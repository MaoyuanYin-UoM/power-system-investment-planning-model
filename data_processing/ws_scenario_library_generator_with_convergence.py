"""
Windstorm Scenario Library Generator with Convergence-Based Stopping

This script generates windstorm scenarios focusing on those affecting the distribution
network (DN), continuing generation until statistical convergence is achieved based on
the Coefficient of Variation (CoV) of Expected Energy Not Supplied (EENS).

Key Features:
- Convergence-based stopping using CoV criterion (Billinton & Li, 1994)
- Filters scenarios to keep only those with DN impact (EENS > 0)
- Adaptive batch sizing for efficient generation
- Compatible output format with existing scenario libraries
"""

import os
import json
import numpy as np
import random
import time
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

# Import project modules
from factories.windstorm_factory import make_windstorm
from factories.network_factory import make_network
from utils import *

# For EENS calculation (simplified OPF)
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


#########################
# EENS CALCULATION
#########################

def calculate_dn_eens_no_investment(
        scenario_data: Dict,
        network,
        solver_name: str = 'gurobi',
        mip_gap: float = 1e-8,
        time_limit: int = 300,
        numeric_focus: int = 2,
        write_lp_on_failure: bool = False,
        write_ilp_on_failure: bool = False,
) -> float:
    """
    Calculate EENS at DN level for a windstorm scenario without any investments.
    Uses existing build_combined_opf_model_under_ws_scenarios method.

    Args:
        scenario_data: Windstorm scenario dictionary with events
        network: Network object instance (from make_network)
        solver_name: Optimization solver to use
        mip_gap: MIP gap for solver
        time_limit: Time limit for solver
        numeric_focus: Numeric focus for Gurobi
        write_lp_on_failure: Write .lp file on failure (default False)
        write_ilp_on_failure: Compute IIS and write .ilp file on failure (default False)

    Returns:
        Total EENS in GWh at DN level

    Raises:
        RuntimeError: If model building or solving fails
    """

    model = None  # Initialize to None for proper cleanup

    try:
        # Build the OPF model using existing method
        model = network.build_combined_opf_model_under_ws_scenarios(
            single_ws_scenario=scenario_data,
            scenario_probability=1.0  # Single scenario
        )

        # Solve the model and get EENS
        eens_mwh = network.solve_combined_opf_model_under_ws_scenarios(
            model=model,
            solver_name=solver_name,
            mip_gap=mip_gap,
            time_limit=time_limit,
            numeric_focus=numeric_focus,
            write_lp_on_failure=write_lp_on_failure,
            write_ilp_on_failure=write_ilp_on_failure,
        )

        # Convert MWh to GWh
        eens_gwh = eens_mwh / 1000.0 if eens_mwh is not None else 0.0

        return eens_gwh

    except Exception as e:
        # Re-raise with more context
        raise RuntimeError(f"EENS calculation failed: {str(e)}")

    finally:
        # Ensure cleanup even if exception occurs
        if model is not None:
            del model
        gc.collect()  # Force garbage collection


#########################
# SCENARIO GENERATION
#########################

def generate_scenario_batch(
        start_idx: int,
        batch_size: int,
        base_seed: int,
        network_preset: str,
        windstorm_preset: str,
        verbose: bool = False
) -> Dict:
    """
    Generate a batch of windstorm scenarios using proper windstorm parameters.
    Adapted directly from ws_scenario_library_generator.py to ensure compatibility.

    Returns:
        Dictionary of scenarios {temp_id: scenario_data}
        Each scenario_data is in the exact format expected by build_combined_opf_model_under_ws_scenarios
    """

    # Load network (shared across batch)
    net = make_network(network_preset)

    # Get network parameters
    num_bus = len(net.data.net.bus)
    num_bch = len(net.data.net.bch)
    num_hrs_prd = 8760  # 1-year period

    # Get branch GIS coordinates
    bch_gis_bgn = []
    bch_gis_end = []
    for bch in net.data.net.bch:
        bgn_idx = net.data.net.bus.index(bch[0])
        end_idx = net.data.net.bus.index(bch[1])
        bgn = (net.data.net.bus_lon[bgn_idx], net.data.net.bus_lat[bgn_idx])
        end = (net.data.net.bus_lon[end_idx], net.data.net.bus_lat[end_idx])
        bch_gis_bgn.append(bgn)
        bch_gis_end.append(end)

    scenarios = {}

    for i in range(batch_size):
        scenario_seed = base_seed + start_idx + i

        # Set seed for reproducibility
        np.random.seed(scenario_seed)
        random.seed(scenario_seed)

        # Create windstorm instance with this seed
        ws = make_windstorm(windstorm_preset)

        # CRITICAL FIX: Initialize windstorm contour distances
        ws.init_ws_path0()  # <-- This line was missing

        # Get number of events for this scenario
        num_events = ws.MC.WS.num_ws_prd[0] if hasattr(ws.MC.WS, 'num_ws_prd') else 1

        # Initialize scenario data (matching exact format from ws_scenario_library_generator.py)
        scenario_data = {
            "scenario_id": f"temp_{start_idx + i:04d}",  # Temporary ID
            "scenario_index": start_idx + i,
            "scenario_seed": scenario_seed,
            "events": []
        }

        # Generate repair times for all branches (shared across events in scenario)
        ttr_min, ttr_max = ws.data.WS.event.ttr
        bch_ttr = np.random.randint(ttr_min, ttr_max + 1, size=num_bch)

        # Generate start/end locations for all events
        start_lons, start_lats, end_lons, end_lats = ws.init_ws_path(num_events)

        # Generate timing for events
        max_lng = max(ws.data.WS.event.lng) + max(ws.data.WS.event.ttr)
        event_start_times = []

        for event_idx in range(num_events):
            lng_ws = ws.MC.WS.lng[event_idx]

            if event_idx == 0:
                # First event can start anywhere (with buffer for repairs)
                max_start = num_hrs_prd - lng_ws - max(bch_ttr)
                ts = np.random.randint(1, max_start + 1) if max_start > 1 else 1
            else:
                # Subsequent events must not overlap and must finish before year end
                min_start = event_start_times[-1] + max_lng
                max_start = num_hrs_prd - lng_ws - max(bch_ttr)
                if min_start < max_start:
                    ts = np.random.randint(min_start, max_start + 1)
                else:
                    # Skip this event if it can't fit
                    continue
            event_start_times.append(ts)

        # Generate each event
        actual_event_idx = 0
        for event_idx in range(len(event_start_times)):
            ts = event_start_times[event_idx]

            # Get duration for this specific event
            lng_ws = ws.MC.WS.lng[event_idx]

            # Get wind speed limits for this event
            lim_v_ws = ws.MC.WS.lim_v_ws_all[event_idx]

            # Create windstorm path using windstorm methods
            path_ws = ws.crt_ws_path(
                start_lons[event_idx], start_lats[event_idx],
                end_lons[event_idx], end_lats[event_idx],
                lng_ws
            )

            # Create windstorm radius over time
            radius_ws = ws.crt_ws_radius(lng_ws)

            # Create wind speed profile
            v_ws = ws.crt_ws_v(lim_v_ws, lng_ws)

            # Generate random numbers for failure probability
            bch_rand_nums = np.random.rand(num_bch, lng_ws)

            # Calculate branch impacts using windstorm's capsule method
            flgs_impacted_bch = ws.compare_capsule_series(
                epicentres=path_ws[:lng_ws],
                radii_km=radius_ws[:lng_ws + 1],  # API allows T+1
                gis_bgn=bch_gis_bgn,
                gis_end=bch_gis_end,
                num_bch=num_bch,
                radius_mode="t"  # Time-varying radius
            )

            # Store event data in exact format expected by OPF model
            event_data = {
                "event_id": int(actual_event_idx + 1),
                "bgn_hr": int(ts),  # Start hour in [1, 8760]
                "duration": int(lng_ws),
                "epicentre": [list(point) for point in path_ws[:lng_ws]],  # List of [lon, lat]
                "radius": radius_ws[:lng_ws].tolist(),  # Hourly radius values
                "gust_speed": v_ws.tolist() if isinstance(v_ws, np.ndarray) else v_ws,  # Wind speeds
                "bch_rand_nums": bch_rand_nums.tolist(),  # Random numbers for failures
                "flgs_impacted_bch": flgs_impacted_bch.tolist(),  # Impact flags
                "bch_ttr": [int(x) for x in bch_ttr]  # Repair times per branch
            }

            scenario_data["events"].append(event_data)
            actual_event_idx += 1

        # Store scenario with temporary ID
        temp_id = f"temp_{start_idx + i:04d}"
        scenarios[temp_id] = scenario_data

        if verbose and (i + 1) % 10 == 0:
            print(f"    Generated {i + 1}/{batch_size} scenarios in batch")

    return scenarios


#########################
# CONVERGENCE FUNCTIONS
#########################

def calculate_convergence_metrics(eens_values: List[float]) -> Dict:
    """
    Calculate convergence metrics for Monte Carlo simulation.

    Reference:
        R. Billinton and W. Li, "Reliability Assessment of Electric Power
        Systems Using Monte Carlo Methods," Plenum Press, 1994,
        Chapter 3, Equations (3.7), (3.8), (3.12).

    Args:
        eens_values: List of EENS values from scenarios

    Returns:
        Dictionary containing:
            - mean: Sample mean (μ)
            - std: Sample standard deviation (σ)
            - cov: Coefficient of variation (σ/μ) - distribution variability
            - conv_beta: Convergence criterion β = σ/(μ√n) - estimate precision
            - n_scenarios: Number of scenarios
            - ci_lower, ci_upper: 95% confidence interval bounds
    """
    if not eens_values:
        return {
            'mean': 0,
            'std': 0,
            'cov': float('inf'),
            'conv_beta': float('inf'),
            'n_scenarios': 0,
            'ci_lower': 0,
            'ci_upper': 0
        }

    n = len(eens_values)
    mean_eens = np.mean(eens_values)
    std_eens = np.std(eens_values, ddof=1) if n > 1 else 0

    # Coefficient of Variation (describes distribution variability)
    cov = std_eens / mean_eens if mean_eens > 0 else float('inf')

    # Convergence Criterion: Coefficient of Variation of the Mean
    # β = σ/(μ√n) - also called relative standard error
    # Billinton & Li (1994) denote this as α in equation (3.7)
    conv_beta = std_eens / (mean_eens * np.sqrt(n)) if mean_eens > 0 and n > 0 else float('inf')

    # 95% Confidence interval
    if n > 1:
        stderr = std_eens / np.sqrt(n)
        ci_lower = mean_eens - 1.96 * stderr
        ci_upper = mean_eens + 1.96 * stderr
    else:
        ci_lower = ci_upper = mean_eens

    return {
        'mean': mean_eens,
        'std': std_eens,
        'cov': cov,  # σ/μ
        'conv_beta': conv_beta,  # σ/(μ√n) ← CONVERGENCE CRITERION
        'n_scenarios': n,
        'ci_lower': max(0, ci_lower),
        'ci_upper': ci_upper,
        'ci_width_ratio': (ci_upper - ci_lower) / mean_eens if mean_eens > 0 else float('inf')
    }


def check_convergence(
        eens_values: List[float],
        threshold: float = 0.05,
        min_scenarios: int = 30
) -> Tuple[bool, Dict]:
    """
    Check Monte Carlo convergence using β criterion.

    Convergence is achieved when:
        β = σ/(μ√n) < threshold

    where β (conv_beta) is the coefficient of variation of the mean estimate,
    also known as the relative standard error.

    Reference:
        R. Billinton and W. Li, "Reliability Assessment of Electric Power
        Systems Using Monte Carlo Methods," Plenum Press, 1994,
        Chapter 3, Section 3.2, Equation (3.7) - denoted as α.

    Args:
        eens_values: List of EENS values
        threshold: Convergence threshold (default 0.05 = 5% precision)
        min_scenarios: Minimum scenarios required (default 30)

    Returns:
        (converged: bool, metrics: dict)
    """
    metrics = calculate_convergence_metrics(eens_values)

    if metrics['n_scenarios'] < min_scenarios:
        return False, metrics

    # Check convergence using β (conv_beta)
    converged = metrics['conv_beta'] < threshold

    return converged, metrics


def determine_next_batch_size(
        current_metrics: Dict,
        hit_rate: float,
        current_batch_size: int,
        n_dn_scenarios: int
) -> int:
    """
    Adaptively determine next batch size based on current statistics.

    Strategy:
    - Low hit rate → larger batches to find DN impacts faster
    - High CoV → moderate batches for exploration
    - Near convergence → smaller batches for fine-tuning
    """

    # If very few DN hits, use large batches
    if hit_rate < 0.05:
        return min(100, current_batch_size * 2)

    # If hit rate is reasonable, adjust based on CoV
    cov = current_metrics.get('cov', float('inf'))

    if cov > 0.20:
        return 50  # Far from convergence
    elif cov > 0.10:
        return 30  # Moderate distance
    elif cov > 0.05:
        return 20  # Getting close
    else:
        return 10  # Fine-tuning

    return current_batch_size


#########################
# OUTPUT FUNCTIONS
#########################

def save_scenario_library(
        scenarios: Dict,
        metadata: Dict,
        output_path: str
) -> None:
    """
    Save scenario library in standard JSON format.
    """
    library = {
        "metadata": metadata,
        "scenarios": scenarios
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(library, f, indent=2, default=str)  # default=str handles numpy types

    print(f"\nScenario library saved to: {output_path}")


def print_progress_report(
        n_total_generated: int,
        n_dn_scenarios: int,
        metrics: Dict,
        elapsed_time: float,
        verbose: bool = True
) -> None:
    """
    Print formatted progress report to console.
    """
    if not verbose:
        return

    hit_rate = n_dn_scenarios / n_total_generated if n_total_generated > 0 else 0

    print("\n" + "=" * 60)
    print("PROGRESS REPORT")
    print("=" * 60)
    print(f"Total scenarios generated: {n_total_generated}")
    print(f"DN-affecting scenarios: {n_dn_scenarios} ({hit_rate:.1%} hit rate)")

    if metrics['n_scenarios'] > 0:
        print(f"\nConvergence Metrics:")
        print(f"  Mean EENS (DN): {metrics['mean']:.3f} GWh")
        print(f"  Std Dev EENS: {metrics['std']:.3f} GWh")
        print(f"  CoV (distribution): {metrics['cov']:.4f}")
        print(f"  Convergence β: {metrics['conv_beta']:.4f}")  # ← Fixed!

        # Only print CI if it exists
        if 'ci_lower' in metrics and 'ci_upper' in metrics:
            print(f"  95% CI: [{metrics['ci_lower']:.3f}, {metrics['ci_upper']:.3f}] GWh")

    print(f"\nElapsed time: {elapsed_time:.1f} seconds")
    print(f"Generation rate: {n_total_generated / elapsed_time:.1f} scenarios/sec")
    print("=" * 60)


#########################
# MAIN GENERATION FUNCTION
#########################

def generate_windstorm_library_with_convergence(
        network_preset: str,
        windstorm_preset: str,
        convergence_threshold: float = 0.05,
        min_dn_scenarios: int = 20,
        max_dn_scenarios: int = 500,
        max_generation_attempts: int = 5000,
        initial_batch_size: int = 10,
        base_seed: int = 10000,
        output_dir: str = "../Scenario_Database/Scenarios_Libraries/Convergence_Based/",
        library_name: Optional[str] = None,
        verbose: bool = True,
        visualize_scenarios: bool = False,
        write_lp_on_failure: bool = False,
        write_ilp_on_failure: bool = False,
) -> Tuple[str, Dict]:
    """
    Generate DN-affecting windstorm scenarios until convergence is achieved.

    Uses Coefficient of Variation (CoV) of the total EENS in DN as the convergence criterion

    Stops when:
    - Converged: CoV < threshold AND n_dn_scenarios >= min_dn_scenarios
    - OR: n_dn_scenarios >= max_dn_scenarios (safety limit)
    - OR: n_total_generated >= max_generation_attempts (prevent infinite loop)

    Args:
        network_preset: Network configuration name
        windstorm_preset: Windstorm parameter preset name
        convergence_threshold: β threshold for convergence (default 0.05 = 5%)
        min_dn_scenarios: Minimum DN scenarios for valid statistics
        max_dn_scenarios: Maximum DN scenarios to generate
        max_generation_attempts: Maximum total scenarios to try
        initial_batch_size: Size of first batch
        base_seed: Base random seed for reproducibility
        output_dir: Output directory for library file
        library_name: Optional custom name for library
        verbose: Print progress information
        visualize_scenarios: Visualize each scenario after generation (no matter accepted or rejected)
        write_lp_on_failure: Write .lp files for failed models (default False)
        write_ilp_on_failure: Compute IIS and write .ilp files for failed models (default False)

    Returns:
        (output_path, final_metrics):
            - output_path: Path to the saved JSON library file
            - final_metrics: Convergence and statistics metrics
    """

    start_time = time.time()

    # ==================
    # 1. INITIALIZATION
    # ==================

    if verbose:
        print("\n" + "="*70)
        print("CONVERGENCE-BASED WINDSTORM SCENARIO GENERATION")
        print("="*70)
        print(f"Network: {network_preset}")
        print(f"Windstorm preset: {windstorm_preset}")
        print(f"Convergence threshold: {convergence_threshold:.3f}")
        print(f"Target: {min_dn_scenarios}-{max_dn_scenarios} DN scenarios")
        print("="*70)

    # Load network ONCE and reuse for all EENS calculations
    print("Loading network model...")
    network = make_network(network_preset)

    # Identify DN failable branches for quick filtering later
    dn_failable_branch_indices = [
        idx for idx in range(len(network.data.net.bch))
        if (network.data.net.branch_level.get(idx + 1, 'T') == 'D' and
            network.data.net.bch_type[idx] == 1)  # Type 1 = line (failable), Type 0 = transformer
    ]

    if verbose:
        print(f"Identified {len(dn_failable_branch_indices)} failable DN branches for filtering")
        print(f"DN branch indices (0-based): {dn_failable_branch_indices}")

    # Initialize tracking variables
    dn_scenarios = {}  # Scenarios with EENS > 0
    dn_eens_values = []  # EENS values for convergence check
    n_total_generated = 0
    n_dn_scenarios = 0
    n_rejected_no_dn_impact = 0  # Track rejections for statistics
    n_solver_failures = 0
    failed_scenarios_info = []  # Store info about failed scenarios
    current_batch_size = initial_batch_size

    # ============================
    # 2. MAIN GENERATION LOOP
    # ============================

    iteration = 0
    converged = False

    while n_dn_scenarios < max_dn_scenarios and n_total_generated < max_generation_attempts:
        iteration += 1

        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Generating batch of {current_batch_size} scenarios...")

        # Generate batch
        batch_start_time = time.time()
        new_batch = generate_scenario_batch(
            start_idx=n_total_generated,
            batch_size=current_batch_size,
            base_seed=base_seed,
            network_preset=network_preset,
            windstorm_preset=windstorm_preset,
            verbose=False
        )

        batch_gen_time = time.time() - batch_start_time

        # Process batch and calculate EENS
        batch_dn_hits = 0
        if verbose:
            print(f"  Batch generated in {batch_gen_time:.1f}s. Calculating EENS...")

        for temp_id, scenario_data in new_batch.items():
            n_total_generated += 1

            # VISUALIZATION: Show scenario before EENS calculation
            if visualize_scenarios:
                print(f"\n  Visualizing generated scenario #{n_total_generated}...")
                visualize_generated_scenario(
                    scenario_data=scenario_data,
                    network=network,
                    scenario_number=n_total_generated,
                    eens_dn=None,
                    accepted=None
                )

            # QUICK CHECK: Does this scenario contain any affected DN failable branches?
            dn_affected = False

            for event in scenario_data.get('events', []):
                impact_flags = event.get('flgs_impacted_bch', [])

                # Skip if no impact flags (shouldn't happen, but safety check)
                if not impact_flags:
                    continue

                # Check if any DN failable branch is impacted at any timestep
                for dn_idx in dn_failable_branch_indices:
                    # Ensure index is within bounds
                    if dn_idx < len(impact_flags):
                        # impact_flags[dn_idx] is an array of flags across timesteps
                        # Check if this DN branch is affected at ANY timestep
                        if any(impact_flags[dn_idx]):
                            dn_affected = True
                            break

                # If we found DN impact in this event, no need to check other events
                if dn_affected:
                    break

            # Skip EENS calculation if no DN branches are affected
            if not dn_affected:
                n_rejected_no_dn_impact += 1
                if visualize_scenarios and verbose:
                    print(f"  → Scenario #{n_total_generated} REJECTED: No DN failable branches affected")
                elif verbose and n_rejected_no_dn_impact <= 10:  # Show first few rejections
                    print(f"  ✗ Rejected: No DN impact (total rejected: {n_rejected_no_dn_impact})")
                continue  # Skip EENS calculation

            # Calculate EENS at DN level
            try:
                eens_dn = calculate_dn_eens_no_investment(
                    scenario_data,
                    network,
                    solver_name='gurobi',
                    mip_gap=1e-8,
                    time_limit=300,
                    write_lp_on_failure=write_lp_on_failure,
                    write_ilp_on_failure=write_ilp_on_failure,
                )
            except Exception as e:
                n_solver_failures += 1

                # Store detailed failure information
                failure_info = {
                    'scenario_number': n_total_generated,
                    'scenario_seed': scenario_data.get('scenario_seed'),
                    'error_type': type(e).__name__,
                    'error_message': str(e)[:200],  # Truncate long messages
                    'num_events': len(scenario_data.get('events', []))
                }
                failed_scenarios_info.append(failure_info)

                # Log the failure
                if verbose:
                    print(f"  ✗ SOLVER FAILURE for scenario #{n_total_generated}:")
                    print(f"    Error type: {type(e).__name__}")
                    print(f"    Error message: {str(e)[:150]}")
                    if n_solver_failures <= 3:  # Show details for first few failures
                        print(f"    Seed: {scenario_data.get('scenario_seed')}")

                # Visualization for failed scenarios
                if visualize_scenarios:
                    print(f"  → Scenario #{n_total_generated} FAILED (solver error)")
                    visualize_generated_scenario(
                        scenario_data=scenario_data,
                        network=network,
                        scenario_number=n_total_generated,
                        eens_dn=None,
                        accepted=False
                    )

                # Skip this scenario (do NOT treat as zero EENS)
                continue  # Important: skip to next scenario

            # Keep scenario if it affects DN
            if eens_dn > 0:
                scenario_id = f"ws_{n_dn_scenarios:04d}"
                dn_scenarios[scenario_id] = scenario_data
                dn_scenarios[scenario_id]['eens_dn_gwh'] = eens_dn
                dn_eens_values.append(eens_dn)
                n_dn_scenarios += 1
                batch_dn_hits += 1

                if visualize_scenarios:
                    print(f"  → Scenario #{n_total_generated} ACCEPTED with EENS")
                    visualize_generated_scenario(
                        scenario_data=scenario_data,
                        network=network,
                        scenario_number=n_total_generated,
                        eens_dn=eens_dn,
                        accepted=True
                    )

                if verbose and batch_dn_hits <= 5:  # Show first few
                    print(f"  ✓ DN impact found: {scenario_id}, EENS = {eens_dn:.3f} GWh")

            else:
                # Note: This case is when DN branches are affected but DN EENS is still zero
                if visualize_scenarios:
                    print(f"  → Scenario #{n_total_generated} REJECTED: Zero EENS despite DN impact")
                    visualize_generated_scenario(
                        scenario_data=scenario_data,
                        network=network,
                        scenario_number=n_total_generated,
                        eens_dn=0.0,
                        accepted=False
                    )

        # Periodic memory cleanup (every 5 iterations)
        if iteration % 5 == 0:
            gc.collect()
            if verbose:
                print(f"  Memory cleanup performed after iteration {iteration}")

        # Report batch results
        batch_elapsed = time.time() - batch_start_time
        if verbose:
            print(f"  Batch complete: {batch_dn_hits}/{current_batch_size} had DN impact")
            print(f"  Batch processing time: {batch_elapsed:.1f}s")

        # Always calculate and report metrics after each batch
        if n_dn_scenarios > 0:
            # Calculate current metrics
            current_metrics = calculate_convergence_metrics(dn_eens_values)

            if verbose:
                print(f"\n  Current Statistics (after batch {iteration}):")
                print(f"    Total generated: {n_total_generated}")
                print(f"    DN scenarios: {n_dn_scenarios}")
                print(f"    Rejected (no DN impact): {n_rejected_no_dn_impact}")
                print(f"    Solver failures: {n_solver_failures}")
                print(f"    Mean EENS: {current_metrics['mean']:.3f} GWh")
                print(f"    Std Dev: {current_metrics['std']:.3f} GWh")
                print(f"    CoV: {current_metrics['cov']:.4f}")
                print(f"    Convergence criterion β: {current_metrics['conv_beta']:.4f}")

                # Show convergence status
                if n_dn_scenarios < min_dn_scenarios:
                    print(
                        f"    Status: Need {min_dn_scenarios - n_dn_scenarios} more scenarios before convergence check")
                else:
                    if current_metrics['cov'] < convergence_threshold:
                        print(
                            f"    Status: ✓ CoV below threshold ({current_metrics['cov']:.4f} < {convergence_threshold})")
                    else:
                        print(
                            f"    Status: CoV above threshold ({current_metrics['cov']:.4f} > {convergence_threshold})")
        else:
            if verbose:
                print(f"\n  Current Statistics (after batch {iteration}):")
                print(f"    Total generated: {n_total_generated}")
                print(f"    DN scenarios: 0 (no valid scenarios yet)")
                print(f"    Rejected (no DN impact): {n_rejected_no_dn_impact}")

        # Check convergence ONLY if we have enough scenarios
        if n_dn_scenarios >= min_dn_scenarios:
            converged, metrics = check_convergence(
                dn_eens_values,
                threshold=convergence_threshold,
                min_scenarios=min_dn_scenarios
            )

            print(f"  DEBUG: Using threshold={convergence_threshold:.4f}, CoV={metrics['cov']:.4f}")

            # Print progress
            elapsed = time.time() - start_time
            print_progress_report(
                n_total_generated,
                n_dn_scenarios,
                metrics,
                elapsed,
                verbose
            )

            if converged:
                print(f"\n✓ CONVERGED after {n_dn_scenarios} DN scenarios!")
                print(f"  Final CoV: {metrics['cov']:.4f} < {convergence_threshold}")
                break  # Exit loop only if converged
            else:
                print("Not converged, continuing...")
                # Adjust batch size for next iteration
                hit_rate = n_dn_scenarios / n_total_generated
                current_batch_size = determine_next_batch_size(
                    metrics, hit_rate, current_batch_size, n_dn_scenarios
                )

                if verbose:
                    print(f"  Next batch size: {current_batch_size}")
                    print(f"  Continuing generation (CoV={metrics['cov']:.4f} > {convergence_threshold})...")

        # Check if reached limits
        if n_dn_scenarios >= max_dn_scenarios:
            if verbose:
                print(f"\n⚠ Reached maximum DN scenarios ({max_dn_scenarios})")
            break

        if n_total_generated >= max_generation_attempts - current_batch_size:
            if verbose:
                print(f"\n⚠ Approaching generation limit, stopping")
            break

    # ====================
    # 3. FINALIZATION AND SAVING
    # ====================

    # Calculate final metrics
    final_metrics = calculate_convergence_metrics(dn_eens_values)
    final_metrics['converged'] = converged
    final_metrics['total_generated'] = n_total_generated
    final_metrics['dn_scenarios'] = n_dn_scenarios
    final_metrics['rejected_no_dn_impact'] = n_rejected_no_dn_impact  # NEW
    final_metrics['eens_calculations_performed'] = n_total_generated - n_rejected_no_dn_impact  # NEW
    final_metrics['hit_rate'] = n_dn_scenarios / n_total_generated if n_total_generated > 0 else 0
    final_metrics['efficiency_gain'] = n_rejected_no_dn_impact / n_total_generated if n_total_generated > 0 else 0

    if verbose:
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Total scenarios generated: {n_total_generated}")
        print(f"  - Rejected (no DN impact): {n_rejected_no_dn_impact}")
        print(f"  - EENS calculations performed: {n_total_generated - n_rejected_no_dn_impact}")
        print(f"  - Scenarios with non-zero EENS: {n_dn_scenarios}")
        print(f"Efficiency gain from quick check: {final_metrics['efficiency_gain']:.1%}")
        print(f"Convergence status: {'CONVERGED' if converged else 'NOT CONVERGED'}")
        if converged:
            print(f"  Final CoV: {final_metrics['cov']:.4f} (threshold: {convergence_threshold:.4f})")
        print("=" * 70 + "\n")

    # Prepare complete library structure (same format as ws_scenario_library_generator.py)
    library = {
        "metadata": {
            "library_type": "windstorm_scenarios_convergence_based",
            "network_preset": network_preset,
            "windstorm_preset": windstorm_preset,
            "num_scenarios": n_dn_scenarios,
            "base_seed": base_seed,
            "generation_date": datetime.now().isoformat(),
            "scenario_id_format": "ws_XXXX (4-digit zero-padded)",

            "convergence_info": {
                "method": "coefficient_of_variation_of_mean",
                "criterion_symbol": "β",
                "criterion_formula": "σ/(μ√n)",
                "threshold": convergence_threshold,
                "converged": converged,
                "final_conv_beta": final_metrics['conv_beta'],
                "final_cov": final_metrics['cov'],
                "min_scenarios_required": min_dn_scenarios
            },

            "generation_statistics": {
                "total_scenarios_generated": n_total_generated,
                "dn_scenarios_kept": n_dn_scenarios,
                "hit_rate": final_metrics['hit_rate'],
                "generation_time_seconds": time.time() - start_time
            },

            "eens_statistics": {
                "mean_gwh": final_metrics['mean'],
                "std_gwh": final_metrics['std'],
                "min_gwh": min(dn_eens_values) if dn_eens_values else 0,
                "max_gwh": max(dn_eens_values) if dn_eens_values else 0,
                "ci_95_lower": final_metrics['ci_lower'],
                "ci_95_upper": final_metrics['ci_upper'],
                "quantiles": {
                    "q10": np.percentile(dn_eens_values, 10) if dn_eens_values else 0,
                    "q25": np.percentile(dn_eens_values, 25) if dn_eens_values else 0,
                    "q50": np.percentile(dn_eens_values, 50) if dn_eens_values else 0,
                    "q75": np.percentile(dn_eens_values, 75) if dn_eens_values else 0,
                    "q90": np.percentile(dn_eens_values, 90) if dn_eens_values else 0
                }
            },

            "note": "Each scenario includes 'eens_dn_gwh' field. Zero-EENS scenarios were filtered out."
        },

        "scenarios": dn_scenarios  # ALL THE ACTUAL SCENARIOS WITH EENS VALUES!
    }

    # Add scenario_probabilities into the metadata (for compatibility with investment model)
    scenario_probabilities = {}
    equal_prob = 1.0 / n_dn_scenarios
    for scenario_id in dn_scenarios.keys():
        scenario_probabilities[scenario_id] = equal_prob

    library["scenario_probabilities"] = scenario_probabilities

    # Generate output filename
    if library_name:
        filename = library_name
        if not filename.endswith('.json'):
            filename += '.json'
    else:
        # Auto-generate descriptive filename
        filename = f"ws_library_{network_preset}_convergence_cov{convergence_threshold:.3f}_{n_dn_scenarios}scenarios.json"

    output_path = os.path.join(output_dir, filename)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # SAVE THE COMPLETE LIBRARY TO DISK
    with open(output_path, 'w') as f:
        json.dump(library, f, indent=2, default=str)  # default=str handles numpy types

    # Verify file was saved and get size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    if verbose:
        print(f"\n✓ Scenario library saved to: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Contains {n_dn_scenarios} scenarios with EENS values")

    # Final cleanup of the network instance
    del network
    gc.collect()

    return output_path, final_metrics


# ====================
# HELPER FUNCTIONS
# ====================

def _generate_representative_library_filename(
        source_metadata: Dict,
        n_representatives: int,
        n_source_scenarios: int,
        selection_method: str = 'quantile'
) -> str:
    """
    Generate standardized filename for representative library.

    Format: rep_scn{N}_{method}_from{M}scn_{net}_{ws}_seed{seed}_beta{threshold}.json

    Examples:
        rep_scn5_quantile_from985scn_29BusGB-Kearsley_29GB_seed10000_beta0.020.json
        rep_scn10_interval_from985scn_29BusGB-Kearsley_29GB_seed10000_beta0.020.json

    Args:
        source_metadata: Metadata dict from source convergence-based library
        n_representatives: Number of representative scenarios
        n_source_scenarios: Number of scenarios in source library
        selection_method: Selection method ('quantile' or 'equal_interval')

    Returns:
        Generated filename string
    """
    # Extract info from source metadata
    network_preset = source_metadata.get('network_preset', 'unknown')
    windstorm_preset = source_metadata.get('windstorm_preset', 'unknown')
    base_seed = source_metadata.get('base_seed', 0)

    # Get convergence threshold (beta)
    convergence_info = source_metadata.get('convergence_info', {})
    threshold = convergence_info.get('threshold', 0.0)

    # Get aliases (these helper functions should exist in the original script)
    net_alias = _get_network_alias(network_preset)
    ws_alias = _get_windstorm_alias(windstorm_preset)

    # Method abbreviation
    method_abbr = 'quantile' if selection_method == 'quantile' else 'interval'

    # Build filename
    filename = (
        f"rep_scn{n_representatives}_"
        f"{method_abbr}_"
        f"from{n_source_scenarios}scn_"
        f"{net_alias}_{ws_alias}_"
        f"seed{base_seed}_"  # FIXED: changed 's' to 'seed'
        f"beta{threshold:.3f}.json"
    )

    return filename


def _get_network_alias(network_preset: str) -> str:
    """
    Get short alias for network preset.

    NOTE: This function should already exist in the original script.
    If not, you'll need to implement it based on your network naming conventions.
    """
    # TODO: Ensure this matches the implementation in your original script
    alias_map = {
        '29_bus_GB_transmission_network_with_Kearsley_GSP_group': '29BusGB-Kearsley',
        # Add other network presets as needed
    }
    return alias_map.get(network_preset, network_preset[:20])


def _get_windstorm_alias(windstorm_preset: str) -> str:
    """
    Get short alias for windstorm preset.

    NOTE: This function should already exist in the original script.
    If not, you'll need to implement it based on your windstorm naming conventions.
    """
    # TODO: Ensure this matches the implementation in your original script
    alias_map = {
        'windstorm_29_bus_GB_transmission_network': '29GB',
        # Add other windstorm presets as needed
    }
    return alias_map.get(windstorm_preset, windstorm_preset[:20])


def _select_representatives_by_quantile(
        scenario_list: List[Tuple],
        eens_values: List[float],
        n_representatives: int,
        eens_field: str,
        verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Select representative scenarios by dividing PDF into equal-probability quantiles.

    This method:
    1. Divides the sorted scenario list into n equal-probability quantiles
    2. Selects the middle scenario from each quantile as representative
    3. Assigns equal probability (1/n) to each representative

    Args:
        scenario_list: List of (original_id, eens_value, scenario_data) tuples (sorted by EENS)
        eens_values: Sorted list of EENS values (corresponding to scenario_list)
        n_representatives: Number of representative scenarios to select
        eens_field: Field name for EENS in scenario data ('eens_dn_gwh' or 'eens_dn')
        verbose: Print selection details

    Returns:
        Tuple of (representatives, selection_info):
            - representatives: Dict with rep_id as key, containing:
                - original_id: Original scenario ID
                - scenario_data: Full scenario data dict
                - eens_gwh: EENS value
                - probability: Representative probability (1/n for quantile method)
                - quantile_info: Dict with quantile-specific information
            - selection_info: Dict with method metadata and selection details
    """

    n_scenarios = len(scenario_list)

    if verbose:
        print(f"\nDividing into {n_representatives} equal-probability quantiles...")

    # Calculate quantile size (number of scenarios per quantile)
    quantile_size = n_scenarios / n_representatives

    # Calculate quantile boundaries (EENS values at quantile splits)
    quantile_percentiles = np.linspace(0, 100, n_representatives + 1)
    quantile_boundaries = [np.percentile(eens_values, p) for p in quantile_percentiles]

    if verbose:
        print(f"Quantile boundaries (EENS in GWh):")
        for i, boundary in enumerate(quantile_boundaries):
            print(f"  Q{i} ({quantile_percentiles[i]:.1f}%): {boundary:.3f} GWh")

    # Select middle scenario from each quantile
    representatives = {}
    selection_info = {
        'quantiles': [],
        'method': 'quantile',
        'n_representatives': n_representatives,
        'quantile_boundaries': quantile_boundaries,
        'quantile_percentiles': quantile_percentiles.tolist()
    }

    for i in range(n_representatives):
        # Determine quantile boundaries (indices in the sorted list)
        start_idx = int(i * quantile_size)
        end_idx = int((i + 1) * quantile_size)

        # Select middle scenario in this quantile
        middle_idx = (start_idx + end_idx) // 2

        original_id, eens_value, scenario_data = scenario_list[middle_idx]

        # Create representative ID
        rep_id = f"rep_{i:02d}"

        # Calculate relative probability (equal for all representatives in quantile method)
        relative_prob = 1.0 / n_representatives

        # Store quantile info
        quantile_info = {
            'quantile_number': i + 1,
            'quantile_range_gwh': [eens_values[start_idx], eens_values[min(end_idx, len(eens_values) - 1)]],
            'percentile_range': [quantile_percentiles[i], quantile_percentiles[i + 1]],
            'scenarios_in_quantile': end_idx - start_idx,
            'selected_index': middle_idx
        }

        # Store representative info
        representatives[rep_id] = {
            'original_id': original_id,
            'scenario_data': scenario_data,
            'eens_gwh': scenario_data[eens_field],
            'probability': relative_prob,
            'quantile_info': quantile_info
        }

        # Add to selection_info
        selection_info['quantiles'].append({
            'rep_id': rep_id,
            'original_id': original_id,
            'eens_gwh': eens_value,
            'quantile': i + 1,
            'quantile_range': quantile_info['quantile_range_gwh']
        })

    if verbose:
        print(f"\nSelected {n_representatives} representative scenarios:")
        for info in selection_info['quantiles']:
            print(f"  {info['rep_id']}: {info['original_id']}, "
                  f"EENS = {info['eens_gwh']:.3f} GWh (Q{info['quantile']})")

    return representatives, selection_info


def _select_representatives_by_equal_intervals(
        scenario_list: List[Tuple],
        eens_values: List[float],
        n_representatives: int,
        eens_field: str,
        eens_min: Optional[float] = None,
        eens_max: Optional[float] = None,
        handle_empty_intervals: str = 'skip',
        verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Select representative scenarios by dividing EENS range into equal-width intervals.

    This method:
    1. Divides the EENS range into n equal-width intervals
    2. For each interval, selects the scenario closest to the interval centre
    3. Assigns probability based on the number of scenarios in each interval

    Args:
        scenario_list: List of (original_id, eens_value, scenario_data) tuples (sorted by EENS)
        eens_values: Sorted list of EENS values (corresponding to scenario_list)
        n_representatives: Number of representative scenarios to select
        eens_field: Field name for EENS in scenario data ('eens_dn_gwh' or 'eens_dn')
        eens_min: Minimum EENS for interval range (auto-detected from data if None)
        eens_max: Maximum EENS for interval range (auto-detected from data if None)
        handle_empty_intervals: How to handle empty intervals:
            - 'skip': Skip empty intervals (results in fewer representatives)
            - 'nearest': Use the nearest scenario from adjacent intervals
        verbose: Print selection details

    Returns:
        Tuple of (representatives, selection_info):
            - representatives: Dict with rep_id as key, containing:
                - original_id: Original scenario ID
                - scenario_data: Full scenario data dict
                - eens_gwh: EENS value
                - probability: Probability based on interval population
                - interval_info: Dict with interval-specific information
            - selection_info: Dict with method metadata and selection details
    """

    n_scenarios = len(scenario_list)

    # Determine EENS range
    eens_min_actual = eens_min if eens_min is not None else min(eens_values)
    eens_max_actual = eens_max if eens_max is not None else max(eens_values)

    if verbose:
        print(f"\nDividing EENS range into {n_representatives} equal-width intervals...")
        print(f"EENS range: [{eens_min_actual:.3f}, {eens_max_actual:.3f}] GWh")

    # Create equal-width intervals
    interval_edges = np.linspace(eens_min_actual, eens_max_actual, n_representatives + 1)
    interval_width = (eens_max_actual - eens_min_actual) / n_representatives

    if verbose:
        print(f"Interval width: {interval_width:.3f} GWh")
        print(f"\nInterval boundaries (EENS in GWh):")
        for i in range(n_representatives + 1):
            print(f"  Edge {i}: {interval_edges[i]:.3f} GWh")

    # Select representative from each interval
    representatives = {}
    selection_info = {
        'intervals': [],
        'method': 'equal_interval',
        'n_representatives': n_representatives,
        'eens_range': [eens_min_actual, eens_max_actual],
        'interval_width': interval_width,
        'interval_edges': interval_edges.tolist(),
        'empty_intervals_skipped': []
    }

    actual_rep_count = 0  # Track actual number of representatives (may differ if intervals are skipped)

    for i in range(n_representatives):
        # Define interval boundaries
        interval_lower = interval_edges[i]
        interval_upper = interval_edges[i + 1]
        interval_centre = (interval_lower + interval_upper) / 2.0

        # Find all scenarios in this interval
        # Use <= for upper bound on the last interval to include the maximum value
        if i == n_representatives - 1:
            scenarios_in_interval = [
                (idx, orig_id, eens_val, scn_data)
                for idx, (orig_id, eens_val, scn_data) in enumerate(scenario_list)
                if interval_lower <= eens_val <= interval_upper
            ]
        else:
            scenarios_in_interval = [
                (idx, orig_id, eens_val, scn_data)
                for idx, (orig_id, eens_val, scn_data) in enumerate(scenario_list)
                if interval_lower <= eens_val < interval_upper
            ]

        n_in_interval = len(scenarios_in_interval)

        # Handle empty intervals
        if n_in_interval == 0:
            if handle_empty_intervals == 'skip':
                if verbose:
                    print(f"\nInterval {i + 1} [{interval_lower:.3f}, {interval_upper:.3f}] GWh: EMPTY - Skipping")
                selection_info['empty_intervals_skipped'].append({
                    'interval_number': i + 1,
                    'interval_range': [interval_lower, interval_upper]
                })
                continue

            elif handle_empty_intervals == 'nearest':
                # Find the nearest scenario to interval centre
                if verbose:
                    print(
                        f"\nInterval {i + 1} [{interval_lower:.3f}, {interval_upper:.3f}] GWh: EMPTY - Using nearest scenario")

                # Find scenario closest to interval centre from entire scenario list
                closest_idx = min(
                    range(len(scenario_list)),
                    key=lambda idx: abs(scenario_list[idx][1] - interval_centre)
                )
                idx_in_list, original_id, eens_value, scenario_data = (
                    closest_idx,
                    scenario_list[closest_idx][0],
                    scenario_list[closest_idx][1],
                    scenario_list[closest_idx][2]
                )

                # Set probability to zero for empty intervals with nearest neighbor
                relative_prob = 0.0
            else:
                raise ValueError(f"Invalid handle_empty_intervals option: {handle_empty_intervals}. "
                                 f"Must be 'skip' or 'nearest'.")
        else:
            # Find scenario closest to interval centre
            closest_in_interval = min(
                scenarios_in_interval,
                key=lambda x: abs(x[2] - interval_centre)  # x[2] is eens_value
            )
            idx_in_list, original_id, eens_value, scenario_data = closest_in_interval

            # Calculate probability based on population in this interval
            relative_prob = n_in_interval / n_scenarios

            if verbose:
                print(f"\nInterval {i + 1} [{interval_lower:.3f}, {interval_upper:.3f}] GWh:")
                print(f"  Scenarios in interval: {n_in_interval}")
                print(f"  Interval centre: {interval_centre:.3f} GWh")
                print(f"  Selected scenario: {original_id}, EENS = {eens_value:.3f} GWh")
                print(f"  Distance to centre: {abs(eens_value - interval_centre):.3f} GWh")
                print(f"  Probability: {relative_prob:.4f}")

        # Create representative ID
        rep_id = f"rep_{actual_rep_count+1:02d}"  # +1 to start from 01
        actual_rep_count += 1

        # Store interval info
        interval_info = {
            'interval_number': i + 1,
            'interval_range_gwh': [interval_lower, interval_upper],
            'interval_centre_gwh': interval_centre,
            'scenarios_in_interval': n_in_interval,
            'selected_index': idx_in_list,
            'distance_to_centre': abs(eens_value - interval_centre)
        }

        # Store representative info
        representatives[rep_id] = {
            'original_id': original_id,
            'scenario_data': scenario_data,
            'eens_gwh': scenario_data[eens_field],
            'probability': relative_prob,
            'interval_info': interval_info
        }

        # Add to selection_info
        selection_info['intervals'].append({
            'rep_id': rep_id,
            'original_id': original_id,
            'eens_gwh': eens_value,
            'interval': i + 1,
            'interval_range': [interval_lower, interval_upper],
            'probability': relative_prob,
            'n_scenarios': n_in_interval
        })

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Selected {actual_rep_count} representative scenarios")
        if len(selection_info['empty_intervals_skipped']) > 0:
            print(f"Skipped {len(selection_info['empty_intervals_skipped'])} empty intervals")
        print(f"Total probability: {sum(rep['probability'] for rep in representatives.values()):.6f}")
        print(f"{'=' * 70}")

    return representatives, selection_info


def generate_representative_ws_scenarios(
        library_path: str,
        n_representatives: int = 5,
        selection_method: str = 'quantile',
        # Parameters for equal_interval method only
        eens_min: Optional[float] = None,
        eens_max: Optional[float] = None,
        handle_empty_intervals: str = 'skip',
        # Common parameters
        output_dir: str = "../Scenario_Database/Scenarios_Libraries/Representatives_from_Convergence_Based/",
        save_library: bool = True,
        library_name: Optional[str] = None,
        verbose: bool = True,
        visualize_pdf: bool = False,
        save_pdf_plot: bool = False,
) -> Tuple[str, Dict]:
    """
    Generate representative windstorm scenarios from convergence-based library.

    This function supports two representative scenario selection methods:

    1. **'quantile'** (Equal-probability quantile-based):
       - Divides the PDF into n equal-probability quantiles
       - Selects the middle scenario from each quantile
       - Each representative has equal probability (1/n)
       - Better for optimization where all quantiles have equal importance

    2. **'equal_interval'** (Equal-width EENS interval-based):
       - Divides the EENS range into n equal-width intervals
       - Selects the scenario closest to each interval centre
       - Probability is proportional to the number of scenarios in each interval
       - Better representation of the actual distribution shape, especially in tail regions

    Args:
        library_path: Path to convergence-based scenario library JSON file
        n_representatives: Number of representative scenarios to select
        selection_method: Selection method ('quantile' or 'equal_interval')

        # equal_interval method parameters
        eens_min: [equal_interval only] Minimum EENS for interval range (auto-detect if None)
        eens_max: [equal_interval only] Maximum EENS for interval range (auto-detect if None)
        handle_empty_intervals: [equal_interval only] How to handle empty intervals:
            - 'skip': Skip empty intervals (may result in fewer representatives)
            - 'nearest': Use nearest scenario from entire set (probability = 0)

        # Common parameters
        output_dir: Directory to save representative library JSON file
        save_library: Whether to save the library to file
        library_name: Custom output filename (auto-generated if None)
        verbose: Print detailed progress and statistics
        visualize_pdf: If True, plot the EENS PDF with representative scenarios
        save_pdf_plot: If True, save the PDF visualization to file

    Returns:
        Tuple of (output_path, selection_info):
            - output_path: Path to the saved representative library JSON file
            - selection_info: Dict containing selection statistics and metadata

    Raises:
        ValueError: If selection_method is not valid or if parameters are invalid
        FileNotFoundError: If library_path does not exist
    """

    # ====================
    # 0. VALIDATE INPUTS
    # ====================

    valid_methods = ['quantile', 'equal_interval']
    if selection_method not in valid_methods:
        raise ValueError(
            f"Invalid selection_method '{selection_method}'. "
            f"Must be one of {valid_methods}"
        )

    if handle_empty_intervals not in ['skip', 'nearest']:
        raise ValueError(
            f"Invalid handle_empty_intervals '{handle_empty_intervals}'. "
            f"Must be 'skip' or 'nearest'"
        )

    if verbose:
        print("\n" + "=" * 70)
        print(f"REPRESENTATIVE SCENARIO SELECTION ({selection_method.upper()})")
        print("=" * 70)
        print(f"Source library: {library_path}")
        print(f"Selection method: {selection_method}")
        print(f"Target representatives: {n_representatives}")
        if selection_method == 'equal_interval':
            print(f"EENS range: [{eens_min if eens_min else 'auto'}, {eens_max if eens_max else 'auto'}]")
            print(f"Empty interval handling: {handle_empty_intervals}")

    # ====================
    # 1. LOAD LIBRARY
    # ====================

    import json
    import os

    with open(library_path, 'r') as f:
        source_library = json.load(f)

    source_scenarios = source_library['scenarios']
    source_metadata = source_library['metadata']

    n_source_scenarios = len(source_scenarios)

    if verbose:
        print(f"\nLoaded source library with {n_source_scenarios} scenarios")

    # ====================
    # 2. SORT SCENARIOS BY EENS
    # ====================

    # Detect EENS field name (backward compatibility)
    first_scenario = next(iter(source_scenarios.values()))
    if 'eens_dn_gwh' in first_scenario:
        eens_field = 'eens_dn_gwh'
    elif 'eens_dn' in first_scenario:
        eens_field = 'eens_dn'
    else:
        raise ValueError(
            "Cannot find EENS field in scenarios. "
            "Expected either 'eens_dn_gwh' or 'eens_dn' field."
        )

    # Create list of (original_id, eens_value, scenario_data) and sort by EENS
    scenario_list = [
        (scenario_id, scenario_data[eens_field], scenario_data)
        for scenario_id, scenario_data in source_scenarios.items()
    ]
    scenario_list.sort(key=lambda x: x[1])  # Sort by EENS value

    # Extract sorted EENS values
    eens_values = [item[1] for item in scenario_list]

    if verbose:
        print(f"EENS statistics:")
        print(f"  Min: {min(eens_values):.3f} GWh")
        print(f"  Max: {max(eens_values):.3f} GWh")
        print(f"  Mean: {np.mean(eens_values):.3f} GWh")
        print(f"  Std: {np.std(eens_values):.3f} GWh")

    # ====================
    # 3. SELECT REPRESENTATIVES (DISPATCH TO METHOD-SPECIFIC FUNCTION)
    # ====================

    if selection_method == 'quantile':
        representatives, selection_info = _select_representatives_by_quantile(
            scenario_list=scenario_list,
            eens_values=eens_values,
            n_representatives=n_representatives,
            eens_field=eens_field,
            verbose=verbose
        )

    elif selection_method == 'equal_interval':
        representatives, selection_info = _select_representatives_by_equal_intervals(
            scenario_list=scenario_list,
            eens_values=eens_values,
            n_representatives=n_representatives,
            eens_field=eens_field,
            eens_min=eens_min,
            eens_max=eens_max,
            handle_empty_intervals=handle_empty_intervals,
            verbose=verbose
        )

    # ====================
    # 4. BUILD REPRESENTATIVE LIBRARY
    # ====================

    # Create new library structure
    representative_library = {
        'scenarios': {},
        'metadata': {}
    }

    # Add representative scenarios
    for rep_id, rep_info in representatives.items():
        representative_library['scenarios'][rep_id] = rep_info['scenario_data']

    # Build metadata
    representative_library['metadata'] = {
        'library_type': f'representative_{selection_method}',
        'n_representatives': len(representatives),
        'selection_method': selection_method,
        'source_library': {
            'path': os.path.basename(library_path),
            'num_scenarios': n_source_scenarios,
            'network_preset': source_metadata.get('network_preset'),
            'windstorm_preset': source_metadata.get('windstorm_preset'),
            'base_seed': source_metadata.get('base_seed'),
            'convergence_info': source_metadata.get('convergence_info')
        },
        'representative_info': selection_info,
        'eens_statistics': {
            'original_mean_gwh': np.mean(eens_values),
            'original_std_gwh': np.std(eens_values),
            'representative_mean_gwh': sum(
                rep['probability'] * rep['eens_gwh']
                for rep in representatives.values()
            ),
            'representative_std_gwh': np.sqrt(sum(
                rep['probability'] * (rep['eens_gwh'] - sum(
                    r['probability'] * r['eens_gwh'] for r in representatives.values()
                )) ** 2
                for rep in representatives.values()
            )),
        },
        'probabilities': {
            rep_id: rep_info['probability']
            for rep_id, rep_info in representatives.items()
        }
    }

    if verbose:
        print(f"\nRepresentative library statistics:")
        print(f"  Number of representatives: {len(representatives)}")
        print(f"  Total probability: {sum(rep['probability'] for rep in representatives.values()):.6f}")
        print(
            f"  Original mean EENS: {representative_library['metadata']['eens_statistics']['original_mean_gwh']:.3f} GWh")
        print(
            f"  Representative mean EENS: {representative_library['metadata']['eens_statistics']['representative_mean_gwh']:.3f} GWh")

    # ====================
    # 5. SAVE LIBRARY
    # ====================

    if save_library:
        # Generate filename if not provided
        if library_name is None:
            library_name = _generate_representative_library_filename(
                source_metadata=source_metadata,
                n_representatives=len(representatives),  # Use actual count (may differ if intervals skipped)
                n_source_scenarios=n_source_scenarios,
                selection_method=selection_method
            )

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Full output path
        output_path = os.path.join(output_dir, library_name)

        scenario_probabilities = {
            rep_id: rep_info['probability']
            for rep_id, rep_info in representatives.items()
        }
        representative_library['scenario_probabilities'] = scenario_probabilities

        # Save library
        with open(output_path, 'w') as f:
            json.dump(representative_library, f, indent=2)

        if verbose:
            print(f"\nRepresentative library saved to:")
            print(f"  {output_path}")
    else:
        output_path = None
        if verbose:
            print("\nLibrary not saved (save_library=False)")

    # ====================
    # 6. VISUALIZATION (Optional)
    # ====================

    if visualize_pdf:
        if verbose:
            print("\n" + "=" * 70)
            print("GENERATING PDF VISUALIZATION")
            print("=" * 70)

        # Calculate mean_eens from eens_values
        mean_eens = np.mean(eens_values)

        # Prepare convergence metrics for display
        convergence_info = source_metadata.get('convergence_info', {})
        eens_stats = source_metadata.get('eens_statistics', {})

        display_metrics = {
            'n_scenarios': len(source_scenarios),  # FIXED: was 'scenarios'
            'mean': mean_eens,  # FIXED: now calculated above
            'std': eens_stats.get('std_gwh', np.std(eens_values)),
            'cov': convergence_info.get('final_cov', np.std(eens_values) / mean_eens),
            'conv_beta': convergence_info.get('final_conv_beta', 0)
        }

        # Generate plot save path if requested
        plot_save_path = None
        if save_pdf_plot:
            plot_filename = library_name.replace('.json', '_pdf_visualization.png')
            plot_save_path = os.path.join(output_dir, plot_filename)

        # Extract quantile_boundaries based on selection method
        # FIXED: Handle both methods properly
        quantile_boundaries_viz = None
        if selection_method == 'quantile' and 'quantile_boundaries' in selection_info:
            quantile_boundaries_viz = selection_info['quantile_boundaries']
        elif selection_method == 'equal_interval' and 'interval_edges' in selection_info:
            # For equal_interval, use interval edges as boundaries
            quantile_boundaries_viz = selection_info['interval_edges']

        # Create visualization with customizable elements
        visualize_eens_pdf_with_representatives(
            eens_values=eens_values,
            representatives=representatives,
            quantile_boundaries=quantile_boundaries_viz,  # FIXED: method-specific
            n_representatives=len(representatives),  # FIXED: use actual count
            convergence_metrics=display_metrics,
            save_path=plot_save_path,
            show_plot=True,
            # Optional elements - customize as needed
            show_kde=False,
            show_quantile_boundaries=True,
            show_representatives=True,
            show_representative_labels=True,
            show_mean=False,
            show_median=False,
            show_cdf=False,
            show_stats_box=True,
            n_bins=60,
            kde_bandwidth=0.2,
            # Optional parameters - customize figure appearances
            title=f"DN EENS Probability Distribution with {len(representatives)} Representative Scenarios",
            figsize=(10, 9),
            title_fontsize=17,
            label_fontsize=16,
            legend_fontsize=15,
            tick_fontsize=14,
            stats_fontsize=15,
            rep_label_fontsize=13,
        )

    return output_path, selection_info

# def generate_representative_ws_scenarios_by_splitting_pdf(
#         library_path: str,
#         n_representatives: int = 5,
#         output_dir: str = "../Scenario_Database/Scenarios_Libraries/Representatives_from_Convergence_Based/",
#         save_library: bool = True,
#         library_name: Optional[str] = None,
#         verbose: bool = True,
#         visualize_pdf: bool = False,
#         save_pdf_plot: bool = False,
# ) -> Tuple[str, Dict]:
#     """
#     Generate representative windstorm scenarios using quantile-based PDF splitting.
#
#     This function:
#     1. Loads a convergence-based scenario library with EENS values
#     2. Sorts scenarios by EENS to construct empirical PDF
#     3. Divides into equal-probability quantiles
#     4. Selects middle scenario from each quantile as representative
#     5. Saves as new library with adjusted probabilities (relative, summing to 1.0)
#
#     Args:
#         library_path: Path to convergence-based scenario library JSON
#         n_representatives: Number of representative scenarios to select
#         output_dir: Directory to save representative library
#         library_name: Output filename (auto-generated if None)
#         verbose: Print progress and statistics
#         visualize_pdf: If True, plot the EENS PDF with representative selection
#         save_pdf_plot: If True, save the PDF visualization to file
#
#     Returns:
#         (output_path, selection_info): Path to saved library and selection statistics
#     """
#
#     if verbose:
#         print("\n" + "=" * 70)
#         print("REPRESENTATIVE SCENARIO SELECTION (Quantile-Based)")
#         print("=" * 70)
#         print(f"Source library: {library_path}")
#         print(f"Target representatives: {n_representatives}")
#
#     # ====================
#     # 1. LOAD SOURCE LIBRARY
#     # ====================
#
#     with open(library_path, 'r') as f:
#         full_library = json.load(f)
#
#     scenarios = full_library['scenarios']
#     source_metadata = full_library['metadata']
#
#     if verbose:
#         print(f"Loaded {len(scenarios)} scenarios from library")
#
#     # ====================
#     # 2. SORT BY EENS
#     # ====================
#
#     # Detect which EENS field name is used in the library
#     # Check first scenario to determine format
#     first_scenario = next(iter(scenarios.values()))
#     if 'eens_dn_gwh' in first_scenario:
#         eens_field = 'eens_dn_gwh'
#         library_format = "new"
#     elif 'eens_dn' in first_scenario:
#         eens_field = 'eens_dn'
#         library_format = "old"
#     else:
#         raise ValueError(
#             "Cannot find EENS field in scenarios. "
#             "Expected either 'eens_dn_gwh' or 'eens_dn' field."
#         )
#
#     if verbose:
#         print(f"Detected library format: {library_format} (EENS field: '{eens_field}')")
#
#     # Create list of (scenario_id, eens_value, full_scenario_data)
#     scenario_list = []
#     for sid, sdata in scenarios.items():
#         if eens_field not in sdata:
#             raise ValueError(f"Scenario {sid} missing 'eens_dn' field")
#         scenario_list.append((sid, sdata[eens_field], sdata))
#
#     # Sort by EENS value (ascending)
#     scenario_list.sort(key=lambda x: x[1])
#
#     eens_values = [x[1] for x in scenario_list]
#     min_eens = min(eens_values)
#     max_eens = max(eens_values)
#     mean_eens = np.mean(eens_values)
#
#     if verbose:
#         print(f"\nEENS Statistics:")
#         print(f"  Range: {min_eens:.3f} - {max_eens:.3f} GWh")
#         print(f"  Mean: {mean_eens:.3f} GWh")
#         print(f"  Median: {np.median(eens_values):.3f} GWh")
#
#     # ====================
#     # 3. QUANTILE DIVISION AND SCENARIO SELECTION
#     # ====================
#
#     n_scenarios = len(scenario_list)
#
#     if verbose:
#         print(f"\nDividing into {n_representatives} equal-probability quantiles...")
#
#     # Calculate quantile size
#     quantile_size = n_scenarios / n_representatives
#
#     # Calculate quantile boundaries (EENS values at quantile splits)
#     quantile_percentiles = np.linspace(0, 100, n_representatives + 1)
#     quantile_boundaries = [np.percentile(eens_values, p) for p in quantile_percentiles]
#
#     if verbose:
#         print(f"Quantile boundaries (EENS in GWh):")
#         for i, boundary in enumerate(quantile_boundaries):
#             print(f"  Q{i} ({quantile_percentiles[i]:.1f}%): {boundary:.3f} GWh")
#
#     # Select middle scenario from each quantile
#     representatives = {}
#     selection_info = {
#         'quantiles': [],
#         'method': 'quantile_middle',
#         'n_representatives': n_representatives
#     }
#
#     for i in range(n_representatives):
#         # Determine quantile boundaries (indices)
#         start_idx = int(i * quantile_size)
#         end_idx = int((i + 1) * quantile_size)
#
#         # Select middle scenario in this quantile
#         middle_idx = (start_idx + end_idx) // 2
#
#         original_id, eens_value, scenario_data = scenario_list[middle_idx]
#
#         # Create representative ID
#         rep_id = f"rep_{i:02d}"
#
#         # Calculate relative probability (equal for all representatives)
#         relative_prob = 1.0 / n_representatives
#
#         # Store quantile info
#         quantile_info = {
#             'quantile_number': i + 1,
#             'quantile_range_gwh': [eens_values[start_idx], eens_values[min(end_idx, len(eens_values) - 1)]],
#             'percentile_range': [quantile_percentiles[i], quantile_percentiles[i + 1]],
#             'scenarios_in_quantile': end_idx - start_idx,
#             'selected_index': middle_idx
#         }
#
#         # Store representative info
#         representatives[rep_id] = {
#             'original_id': original_id,
#             'scenario_data': scenario_data,
#             'eens_gwh': scenario_data[eens_field],
#             'probability': relative_prob,
#             'quantile_info': quantile_info
#         }
#
#         # Add to selection_info
#         selection_info['quantiles'].append({
#             'rep_id': rep_id,
#             'original_id': original_id,
#             'eens_gwh': eens_value,
#             'quantile': i + 1,
#             'quantile_range': quantile_info['quantile_range_gwh']
#         })
#
#     if verbose:
#         print(f"\nSelected {n_representatives} representative scenarios:")
#         for info in selection_info['quantiles']:
#             print(f"  {info['rep_id']}: {info['original_id']}, "
#                   f"EENS = {info['eens_gwh']:.3f} GWh (Q{info['quantile']})")
#
#     # ====================
#     # 4. CREATE REPRESENTATIVE LIBRARY
#     # ====================
#
#     # Calculate expected EENS from representatives
#     relative_prob = 1.0 / n_representatives
#     expected_eens = sum(
#         rep['eens_gwh'] * relative_prob
#         for rep in representatives.values()
#     )
#
#     representative_library = {
#         "metadata": {
#             "generation_date": datetime.now().isoformat(),
#             "library_type": "representative_windstorm_scenarios",
#             "network_preset": source_metadata.get('network_preset'),
#             "windstorm_preset": source_metadata.get('windstorm_preset'),
#             "num_representative_scenarios": n_representatives,
#             'base_seed': source_metadata.get('base_seed', None),
#             'equal_probability': True,
#             'probability_per_scenario': 1.0 / n_representatives,
#
#             "source_library": {
#                 "path": library_path,
#                 "num_scenarios": len(scenarios),
#                 "eens_range_gwh": [min_eens, max_eens],
#                 "convergence_info": source_metadata.get('convergence_info', {})
#             },
#
#             "selection_method": {
#                 "method": "quantile_based",
#                 "description": "Equal-probability quantiles with middle scenario selection",
#                 "n_representatives": n_representatives,
#                 "probability_per_representative": 1.0 / n_representatives  # Relative probability
#             },
#
#             "representative_statistics": {
#                 "eens_values_gwh": [rep['eens_gwh'] for rep in representatives.values()],
#                 "probabilities": [rep['probability'] for rep in representatives.values()],
#                 "expected_eens_gwh": expected_eens,
#                 "coverage": {
#                     "min_eens_gwh": min([rep['eens_gwh'] for rep in representatives.values()]),
#                     "max_eens_gwh": max([rep['eens_gwh'] for rep in representatives.values()]),
#                     "captures_range_percent": (
#                         (max([rep['eens_gwh'] for rep in representatives.values()]) -
#                          min([rep['eens_gwh'] for rep in representatives.values()])) /
#                         (max_eens - min_eens) * 100 if max_eens > min_eens else 100
#                     )
#                 }
#             }
#         },
#
#         "scenarios": {}
#     }
#
#     # Add scenarios with adjusted probabilities
#     for rep_id, rep_info in representatives.items():
#         # Copy full scenario data
#         scenario_data = rep_info['scenario_data'].copy()
#
#         # Add metadata about selection (no probability field in individual scenarios)
#         scenario_data['representative_metadata'] = {
#             'original_scenario_id': rep_info['original_id'],
#             'quantile_info': rep_info['quantile_info'],
#             'selection_method': 'quantile_middle',
#             'eens_gwh': rep_info['eens_gwh']
#         }
#
#         # Use representative ID
#         representative_library['scenarios'][rep_id] = scenario_data
#
#     # Add top-level scenario_probabilities (relative probabilities summing to 1.0)
#     scenario_probabilities = {}
#     relative_prob = 1.0 / n_representatives
#
#     for rep_id in representative_library['scenarios'].keys():
#         scenario_probabilities[rep_id] = relative_prob
#
#     representative_library["scenario_probabilities"] = scenario_probabilities
#
#     # ====================
#     # 5. SAVE LIBRARY (Optional)
#     # ====================
#
#     if save_library:
#         # Create output directory
#         os.makedirs(output_dir, exist_ok=True)
#
#         # Generate output filename if not provided
#         if library_name is None:
#             library_name = _generate_representative_library_filename(
#                 source_metadata=source_metadata,
#                 n_representatives=n_representatives,
#                 n_source_scenarios=len(scenarios)
#             )
#
#             if verbose:
#                 print(f"\nAuto-generated filename: {library_name}")
#
#         output_path = os.path.join(output_dir, library_name)
#
#         # Save library to disk
#         with open(output_path, 'w') as f:
#             json.dump(representative_library, f, indent=2)
#             f.flush()
#             os.fsync(f.fileno())
#
#         file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
#
#         if verbose:
#             print(f"\n✓ Representative library saved to:")
#             print(f"  {output_path}")
#             print(f"  File size: {file_size_mb:.2f} MB")
#     else:
#         if verbose:
#             print(f"\n⚠ Library NOT saved to disk (save_to_disk=False)")
#             print(f"  Library data available in returned dictionary")
#
#     # ====================
#     # 5. VISUALIZATION (Optional)
#     # ====================
#
#     if visualize_pdf:
#         if verbose:
#             print("\n" + "=" * 70)
#             print("GENERATING PDF VISUALIZATION")
#             print("=" * 70)
#
#         # Prepare convergence metrics for display
#         convergence_info = source_metadata.get('convergence_info', {})
#         eens_stats = source_metadata.get('eens_statistics', {})
#
#         display_metrics = {
#             'n_scenarios': len(scenarios),
#             'mean': mean_eens,
#             'std': eens_stats.get('std_gwh', np.std(eens_values)),
#             'cov': convergence_info.get('final_cov', np.std(eens_values) / mean_eens),
#             'conv_beta': convergence_info.get('final_conv_beta', 0)
#         }
#
#         # Generate plot save path if requested
#         plot_save_path = None
#         if save_pdf_plot:
#             plot_filename = library_name.replace('.json', '_pdf_visualization.png')
#             plot_save_path = os.path.join(output_dir, plot_filename)
#
#         # Create visualization with customizable elements
#         visualize_eens_pdf_with_representatives(
#             eens_values=eens_values,
#             representatives=representatives,
#             quantile_boundaries=quantile_boundaries,
#             n_representatives=n_representatives,
#             convergence_metrics=display_metrics,
#             title=f"DN EENS Distribution with {n_representatives} Representative Scenarios",
#             save_path=plot_save_path,
#             show_plot=True,
#             # Optional elements - customize as needed
#             show_kde=True,  # KDE smooth curve
#             show_quantile_boundaries=True,  # Orange dashed lines
#             show_representatives=True,  # Red lines for selected scenarios
#             show_representative_labels=False, # Show labels next to each representative line
#             show_mean=False,  # Green mean line
#             show_median=False,  # Purple median line
#             show_cdf=False,  # Bottom CDF subplot
#             show_stats_box=True,  # Statistics text box
#             n_bins=None,  # Number of histogram bins (None for auto)
#             kde_bandwidth=0.2,  # Reduce bandwidth for better fit
#         )
#
#     # ====================
#     # 7. FINAL REPORT
#     # ====================
#
#     if verbose:
#         print("\n" + "=" * 70)
#         print("SELECTION COMPLETE")
#         print("=" * 70)
#         print(f"Representatives selected: {n_representatives}")
#         print(f"Expected EENS (relative): {expected_eens:.3f} GWh")
#         print(f"Each representative probability (relative): {1.0 / n_representatives:.3f}")
#         print(f"\nOutput file: {output_path}")
#         print(f"File size: {file_size_mb:.1f} MB")
#
#         print("\nRepresentative Summary:")
#         for i, (rep_id, rep) in enumerate(representatives.items()):
#             print(f"  {rep_id}: EENS = {rep['eens_gwh']:6.2f} GWh, "
#                   f"P(rel) = {1.0 / n_representatives:.3f}, "
#                   f"From Q{i + 1} (orig: {rep['original_id']})")
#
#     # Prepare return statistics
#     selection_statistics = {
#         'n_source_scenarios': len(scenarios),
#         'n_representatives': n_representatives,
#         'expected_eens_gwh': expected_eens,  # Expected EENS using relative probabilities
#         'eens_range': [min_eens, max_eens],
#         'representative_eens': [rep['eens_gwh'] for rep in representatives.values()],
#         'relative_probabilities': [1.0 / n_representatives] * n_representatives,  # All equal
#         'output_path': output_path,
#         'selection_details': selection_info
#     }
#
#     return output_path, selection_statistics


#########################
# VISUALIZATION FUNCTIONS
#########################

def visualize_generated_scenario(
        scenario_data: Dict,
        network,
        scenario_number: int,
        eens_dn: float = None,
        accepted: bool = None
):
    """
    Visualize a generated windstorm scenario.

    Args:
        scenario_data: The scenario dictionary containing events
        network: Network object
        scenario_number: Sequential number of this generated scenario
        eens_dn: EENS value if calculated (None if not yet calculated)
        accepted: Whether scenario was accepted (True/False/None if not yet determined)
    """
    events = scenario_data.get('events', [])
    if not events:
        print(f"  Scenario {scenario_number} has no events to visualize")
        return

    # Get network GIS data
    network.set_gis_data()
    bch_gis_bgn = network._get_bch_gis_bgn()
    bch_gis_end = network._get_bch_gis_end()
    has_branch_levels = hasattr(network.data.net, 'branch_level')

    # Create figure with subplots for each event
    num_events = len(events)
    fig, axes = plt.subplots(1, num_events, figsize=(8 * num_events, 7))
    if num_events == 1:
        axes = [axes]

    for event_idx, event in enumerate(events):
        ax = axes[event_idx]

        # Extract windstorm data
        epicentres = np.array(event["epicentre"])
        radius_km = event["radius"]

        # Convert radius from km to degrees
        radius_deg = []
        for i, (lon, lat) in enumerate(epicentres):
            lat_factor = 111  # 1 degree latitude ≈ 111 km
            r_deg = radius_km[i] / lat_factor
            radius_deg.append(r_deg)

        # Plot network branches
        tn_plotted = False
        dn_plotted = False

        for idx, (bgn, end) in enumerate(zip(bch_gis_bgn, bch_gis_end)):
            if has_branch_levels:
                branch_level = network.data.net.branch_level.get(idx + 1, 'T')
            else:
                branch_level = 'T'

            if branch_level == 'T' or branch_level == 'T-D':
                color = 'darkgreen'
                label = 'Transmission Branch' if not tn_plotted else ""
                tn_plotted = True
            else:
                color = 'orange'
                label = 'Distribution Branch' if not dn_plotted else ""
                dn_plotted = True

            ax.plot([bgn[0], end[0]], [bgn[1], end[1]], color=color,
                    alpha=0.7, linewidth=1.2, label=label)

        # Plot windstorm path
        ax.plot(epicentres[:, 0], epicentres[:, 1], 'bo-',
                label="Windstorm Path", alpha=0.8, linewidth=2, markersize=5)

        # Plot epicentres and circles for each timestep
        for i, (lon, lat) in enumerate(epicentres):
            ax.scatter(lon, lat, color="blue", s=40, zorder=3)
            circle = Circle((lon, lat), radius_deg[i], color='blue',
                            alpha=0.2, fill=True)
            ax.add_patch(circle)

        # Set axis limits
        bus_lons = network._get_bus_lon()
        bus_lats = network._get_bus_lat()
        xmin, xmax = min(bus_lons), max(bus_lons)
        ymin, ymax = min(bus_lats), max(bus_lats)
        ax.set_xlim(xmin - 1, xmax + 1)
        ax.set_ylim(ymin - 1, ymax + 1)

        # Labels and title
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.tick_params(axis='both', labelsize=10)

        # Title with status information
        title = f"Scenario #{scenario_number}, Event {event_idx + 1}"
        if eens_dn is not None:
            title += f"\nEENS: {eens_dn:.3f} GWh"
        if accepted is not None:
            status = "ACCEPTED" if accepted else "REJECTED"
            title += f" - {status}"

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()


def visualize_eens_pdf_with_representatives(
        eens_values: List[float],
        representatives: Optional[Dict] = None,
        quantile_boundaries: Optional[List[float]] = None,
        n_representatives: int = None,
        convergence_metrics: Optional[Dict] = None,
        title: str = "Distribution of DN EENS from Convergence-Based Scenarios",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
        show_plot: bool = True,
        # Optional elements control
        show_kde: bool = True,
        show_quantile_boundaries: bool = True,
        show_representatives: bool = True,
        show_representative_labels: bool = True,
        show_mean: bool = True,
        show_median: bool = True,
        show_cdf: bool = True,
        show_stats_box: bool = True,
        n_bins: Optional[int] = None,
        kde_bandwidth: Optional[float] = None,
        # NEW: Typography and sizing controls
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        legend_fontsize: int = 10,
        tick_fontsize: int = 10,
        stats_fontsize: int = 11,
        rep_label_fontsize: int = 9,
):
    """
    Visualize the PDF of DN EENS values with quantile divisions and representative scenarios.

    Args:
        eens_values: List of EENS values from all scenarios (sorted or unsorted)
        representatives: Dict of representative scenario info (from quantile selection)
        quantile_boundaries: List of EENS values at quantile boundaries
        n_representatives: Number of representatives (for quantile lines if boundaries not provided)
        convergence_metrics: Dict with convergence statistics (mean, std, cov, etc.)
        title: Plot title
        save_path: Path to save figure (if None, don't save)
        figsize: Figure size (width, height) in inches
        show_plot: Whether to display the plot

        # Display options
        show_kde: Show kernel density estimate (smooth PDF curve)
        show_quantile_boundaries: Show orange dashed lines at quantile boundaries
        show_representatives: Show red lines for representative scenarios
        show_representative_labels: Show text labels for representatives
        show_mean: Show green dotted line for mean EENS
        show_median: Show purple dotted line for median EENS
        show_cdf: Show cumulative distribution function subplot at bottom
        show_stats_box: Show statistics text box (n, μ, σ, CoV, β)
        n_bins: Number of histogram bins (auto-calculated if None)
        kde_bandwidth: KDE bandwidth (None for 'scott', float for custom, or 'silverman')

        # Typography options
        title_fontsize: Font size for main title
        label_fontsize: Font size for axis labels
        legend_fontsize: Font size for legend
        tick_fontsize: Font size for tick labels
        stats_fontsize: Font size for statistics box text
        rep_label_fontsize: Font size for representative scenario labels

    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Sort EENS values for proper visualization
    eens_sorted = sorted(eens_values)
    n_scenarios = len(eens_sorted)

    # Create figure with subplots (conditional on show_cdf)
    if show_cdf:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] * 0.75))
        ax2 = None

    # ====================
    # SUBPLOT 1: Histogram + PDF
    # ====================

    # Plot histogram
    if n_bins is None:
        n_bins = min(50, max(20, n_scenarios // 5))  # Adaptive number of bins

    counts, bin_edges, patches = ax1.hist(
        eens_sorted,
        bins=n_bins,
        density=True,
        alpha=0.6,
        color='steelblue',
        edgecolor='black',
        label='Empirical PDF'
    )

    # Plot KDE (kernel density estimate) for smooth PDF
    if show_kde:
        try:
            from scipy.stats import gaussian_kde

            # Determine bandwidth method
            if kde_bandwidth is None:
                bw_method = 'scott'  # Default
            elif isinstance(kde_bandwidth, str):
                bw_method = kde_bandwidth  # 'scott' or 'silverman'
            else:
                bw_method = float(kde_bandwidth)  # Custom numeric value

            kde = gaussian_kde(eens_sorted, bw_method=bw_method)
            x_kde = np.linspace(min(eens_sorted), max(eens_sorted), 200)
            y_kde = kde(x_kde)

            # Add bandwidth info to label
            if isinstance(bw_method, (int, float)):
                label_text = f'KDE (h={bw_method:.3f})'
            else:
                label_text = f'KDE ({bw_method})'

            ax1.plot(x_kde, y_kde, 'r-', linewidth=2, label=label_text, alpha=0.8)
        except ImportError:
            if show_plot:
                print("Warning: scipy not available, skipping KDE plot")

    # Determine boundary label based on representatives (auto-detect method)
    boundary_label = 'Interval boundary'  # Default
    if representatives is not None and len(representatives) > 0:
        first_rep = next(iter(representatives.values()))
        if 'quantile_info' in first_rep:
            boundary_label = 'Quantile boundary'
        elif 'interval_info' in first_rep:
            boundary_label = 'Interval boundary'

    # Plot quantile/interval boundaries
    if show_quantile_boundaries:
        if quantile_boundaries is not None:
            for i, boundary in enumerate(quantile_boundaries):
                ax1.axvline(
                    boundary,
                    color='orange',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    label=boundary_label if i == 0 else ""
                )
        elif n_representatives is not None:
            # Calculate quantile boundaries if not provided
            quantiles = np.linspace(0, 100, n_representatives + 1)
            boundaries = np.percentile(eens_sorted, quantiles)
            for i, boundary in enumerate(boundaries):
                ax1.axvline(
                    boundary,
                    color='orange',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    label=boundary_label if i == 0 else ""
                )

    # Plot representative scenarios
    if show_representatives and representatives is not None:
        rep_eens = [rep['eens_gwh'] for rep in representatives.values()]
        rep_ids = list(representatives.keys())

        # Mark representatives on histogram
        for i, (rep_id, eens) in enumerate(zip(rep_ids, rep_eens)):
            ax1.axvline(
                eens,
                color='red',
                linestyle=':',
                linewidth=2.0,
                alpha=0.9,
                label='Representative scenario' if i == 0 else ""
            )

            # Add text label for representative
            if show_representative_labels:
                y_max = ax1.get_ylim()[1]
                # Get the relative probability for this representative
                rep_info = representatives[rep_id]
                relative_prob = rep_info.get('probability', 1.0 / len(representatives))
                # Format label with relative probability
                label_text = f'Relative prob. = {relative_prob:.3f}'
                ax1.text(
                    eens,
                    y_max * 0.95,
                    label_text,
                    rotation=90,
                    verticalalignment='top',
                    horizontalalignment='right',
                    fontsize=rep_label_fontsize,
                    fontweight='bold',
                    color='red'
                )

    # Plot statistics lines
    mean_eens = np.mean(eens_sorted)
    median_eens = np.median(eens_sorted)

    if show_mean:
        ax1.axvline(mean_eens, color='green', linestyle=':', linewidth=2,
                    label=f'Mean: {mean_eens:.3f} GWh', alpha=0.8)

    if show_median:
        ax1.axvline(median_eens, color='purple', linestyle=':', linewidth=2,
                    label=f'Median: {median_eens:.3f} GWh', alpha=0.8)

    # Labels and title
    ax1.set_xlabel('EENS (GWh)', fontsize=label_fontsize, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=label_fontsize, fontweight='bold')
    ax1.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=legend_fontsize, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=tick_fontsize)

    # Add convergence metrics text box
    if show_stats_box and convergence_metrics is not None:
        textstr = '\n'.join([
            f"n = {convergence_metrics.get('n_scenarios', n_scenarios)}",
            f"μ = {convergence_metrics.get('mean', mean_eens):.3f} GWh",
            f"σ = {convergence_metrics.get('std', np.std(eens_sorted)):.3f} GWh",
            f"CoV = {convergence_metrics.get('cov', np.std(eens_sorted) / mean_eens):.4f}",
            f"β = {convergence_metrics.get('conv_beta', 0):.4f}"
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.98, 0.5, textstr, transform=ax1.transAxes, fontsize=stats_fontsize,
                 verticalalignment='center', horizontalalignment='right',  # CHANGED alignment
                 bbox=props, family='monospace')

    # ====================
    # SUBPLOT 2: Empirical CDF (Optional)
    # ====================

    if show_cdf and ax2 is not None:
        # Calculate empirical CDF
        cdf_x = np.sort(eens_sorted)
        cdf_y = np.arange(1, len(cdf_x) + 1) / len(cdf_x)

        ax2.plot(cdf_x, cdf_y, 'b-', linewidth=2, label='Empirical CDF')
        ax2.set_xlabel('EENS (GWh)', fontsize=label_fontsize, fontweight='bold')
        ax2.set_ylabel('Cumulative Probability', fontsize=label_fontsize, fontweight='bold')
        ax2.set_title('Cumulative Distribution Function', fontsize=title_fontsize, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=tick_fontsize)

        # Mark quantile boundaries on CDF
        if show_quantile_boundaries and quantile_boundaries is not None:
            quantile_probs = np.linspace(0, 1, len(quantile_boundaries))
            for boundary, prob in zip(quantile_boundaries, quantile_probs):
                ax2.plot(boundary, prob, 'o', color='orange', markersize=8, alpha=0.7)

        # Mark representatives on CDF
        if show_representatives and representatives is not None:
            rep_eens = [rep['eens_gwh'] for rep in representatives.values()]
            for eens in rep_eens:
                # Find CDF value at this EENS
                idx = np.searchsorted(cdf_x, eens)
                if idx < len(cdf_y):
                    ax2.plot(eens, cdf_y[idx], 'o', color='red', markersize=10,
                             alpha=0.9, markeredgewidth=2, markeredgecolor='darkred')

        ax2.legend(loc='lower right', fontsize=legend_fontsize)

    # Tight layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to: {save_path}")

    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig


def visualize_scenario_library_pdf(
        library_path: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
        show_plot: bool = True,
        # Optional elements control
        show_kde: bool = True,
        show_quantile_boundaries: bool = False,
        show_representatives: bool = True,  # Auto-detect if library has representatives
        show_representative_labels: bool = True,
        show_mean: bool = True,
        show_median: bool = True,
        show_cdf: bool = True,
        show_stats_box: bool = True,
        n_bins: Optional[int] = None,
        kde_bandwidth: Optional[float] = None
):
    """
    Visualize the PDF of EENS distribution directly from a scenario library file.

    This is a convenience function that loads a scenario library and visualizes its
    EENS distribution. Works with both convergence-based and representative libraries.

    Args:
        library_path: Path to scenario library JSON file
        title: Plot title (auto-generated if None)
        save_path: Path to save figure (if None, don't save)
        figsize: Figure size (width, height)
        show_plot: Whether to display the plot
        show_kde: Show kernel density estimate (smooth PDF curve)
        show_quantile_boundaries: Show quantile boundaries (only for representative libraries)
        show_representatives: Show representative scenario markers (auto-detected)
        show_representative_labels: Show text labels for representatives
        show_mean: Show mean line
        show_median: Show median line
        show_cdf: Show CDF subplot
        show_stats_box: Show statistics text box
        n_bins: Number of histogram bins (auto if None)
        kde_bandwidth: KDE bandwidth (None for 'scott', float for custom)

    Returns:
        matplotlib figure object
    """

    # ====================
    # 1. LOAD LIBRARY
    # ====================

    with open(library_path, 'r') as f:
        library = json.load(f)

    scenarios = library['scenarios']
    metadata = library['metadata']
    library_type = metadata.get('library_type', 'unknown')

    # ====================
    # 2. EXTRACT EENS VALUES
    # ====================

    # Detect EENS field name (backward compatibility)
    first_scenario = next(iter(scenarios.values()))
    if 'eens_dn_gwh' in first_scenario:
        eens_field = 'eens_dn_gwh'
    elif 'eens_dn' in first_scenario:
        eens_field = 'eens_dn'
    else:
        raise ValueError(
            "Cannot find EENS field in scenarios. "
            "Expected either 'eens_dn_gwh' or 'eens_dn' field."
        )

    # Extract EENS values
    eens_values = [scenario[eens_field] for scenario in scenarios.values()]

    # ====================
    # 3. DETECT LIBRARY TYPE AND EXTRACT RELEVANT INFO
    # ====================

    representatives = None
    quantile_boundaries = None
    n_representatives = None

    # Check if this is a representative library
    is_representative_library = 'representative' in library_type.lower()

    if is_representative_library and show_representatives:
        # Extract representative info
        representatives = {}
        for scenario_id, scenario_data in scenarios.items():
            representatives[scenario_id] = {
                'eens_gwh': scenario_data[eens_field],
                'scenario_data': scenario_data
            }

        n_representatives = len(representatives)

        # Calculate quantile boundaries if requested
        if show_quantile_boundaries:
            # Get source library info if available
            source_info = metadata.get('source_library', {})
            if 'num_scenarios' in source_info:
                # Representative library - calculate quantiles from representatives
                quantile_percentiles = np.linspace(0, 100, n_representatives + 1)
                quantile_boundaries = [np.percentile(eens_values, p) for p in quantile_percentiles]
    else:
        # Convergence-based library - no representatives to show
        if show_representatives:
            show_representatives = False  # Override since no representatives exist

    # ====================
    # 4. PREPARE CONVERGENCE METRICS
    # ====================

    convergence_info = metadata.get('convergence_info', {})
    eens_stats = metadata.get('eens_statistics', {})

    convergence_metrics = {
        'n_scenarios': len(scenarios),
        'mean': eens_stats.get('mean_gwh', np.mean(eens_values)),
        'std': eens_stats.get('std_gwh', np.std(eens_values)),
        'cov': convergence_info.get('final_cov', np.std(eens_values) / np.mean(eens_values)),
        'conv_beta': convergence_info.get('final_conv_beta', 0)
    }

    # ====================
    # 5. GENERATE TITLE
    # ====================

    if title is None:
        if is_representative_library:
            title = f"DN EENS Distribution with {n_representatives} Representative Scenarios"
        else:
            n_scenarios = len(scenarios)
            title = f"DN EENS Distribution from Convergence-Based Scenarios (n={n_scenarios})"

    # ====================
    # 6. CALL VISUALIZATION FUNCTION
    # ====================

    fig = visualize_eens_pdf_with_representatives(
        eens_values=eens_values,
        representatives=representatives,
        quantile_boundaries=quantile_boundaries,
        n_representatives=n_representatives,
        convergence_metrics=convergence_metrics,
        title=title,
        save_path=save_path,
        figsize=figsize,
        show_plot=show_plot,
        show_kde=show_kde,
        show_quantile_boundaries=show_quantile_boundaries,
        show_representatives=show_representatives,
        show_representative_labels=show_representative_labels,
        show_mean=show_mean,
        show_median=show_median,
        show_cdf=show_cdf,
        show_stats_box=show_stats_box,
        n_bins=n_bins,
        kde_bandwidth=kde_bandwidth
    )

    return fig

#########################
# COMPARISON FUNCTIONS
#########################

def compare_original_and_representative_ws_scenarios(
        original_library_path: str,
        representative_library_path: str,
        metrics_to_compare: Union[str, List[str]] = 'all',
        verbose: bool = True
) -> Dict:
    """
    Compare statistical metrics between original and representative scenario sets.

    Now correctly handles weighted statistics for representative scenarios with
    non-uniform probabilities (e.g., from interval-based selection).

    Args:
        original_library_path: Path to original library JSON
        representative_library_path: Path to representative library JSON
        metrics_to_compare: 'all' or list from ['mean', 'std', 'variance', 'range', 'percentiles']
        verbose: Print comparison report

    Returns:
        Dict with 'original', 'representative', 'errors', and 'library_info' keys
    """

    # Parse metrics
    available_metrics = ['mean', 'std', 'variance', 'range', 'percentiles']
    if metrics_to_compare == 'all':
        selected_metrics = available_metrics
    else:
        invalid = [m for m in metrics_to_compare if m not in available_metrics]
        if invalid:
            raise ValueError(f"Invalid metrics: {invalid}. Available: {available_metrics}")
        selected_metrics = metrics_to_compare

    # Load libraries
    with open(original_library_path, 'r') as f:
        orig_lib = json.load(f)
    with open(representative_library_path, 'r') as f:
        rep_lib = json.load(f)

    # Detect EENS field name (backward compatibility)
    first_orig = next(iter(orig_lib['scenarios'].values()))
    eens_field_orig = 'eens_dn_gwh' if 'eens_dn_gwh' in first_orig else 'eens_dn'

    first_rep = next(iter(rep_lib['scenarios'].values()))
    eens_field_rep = 'eens_dn_gwh' if 'eens_dn_gwh' in first_rep else 'eens_dn'

    # Extract EENS values
    eens_orig = np.array([s[eens_field_orig] for s in orig_lib['scenarios'].values()])

    # Extract representative EENS values AND probabilities
    rep_scenarios = rep_lib['scenarios']
    rep_probabilities_dict = rep_lib.get('scenario_probabilities', {})

    # Build aligned arrays for representative scenarios
    eens_rep_list = []
    prob_rep_list = []
    for scenario_id, scenario_data in rep_scenarios.items():
        eens_rep_list.append(scenario_data[eens_field_rep])
        # Get probability (default to equal probability if not found)
        prob_rep_list.append(rep_probabilities_dict.get(scenario_id, 1.0 / len(rep_scenarios)))

    eens_rep = np.array(eens_rep_list)
    prob_rep = np.array(prob_rep_list)

    # Normalize probabilities (should sum to 1.0, but ensure it)
    prob_rep = prob_rep / prob_rep.sum()

    # Initialize results
    results = {
        'library_info': {
            'n_original': len(eens_orig),
            'n_representative': len(eens_rep),
            'reduction_factor': len(eens_orig) / len(eens_rep),
            'representative_uses_weighted_stats': True  # Flag to indicate weighted calculations
        },
        'original': {},
        'representative': {},
        'errors': {}
    }

    # Calculate metrics
    if 'mean' in selected_metrics:
        # Original: unweighted mean (equal probabilities)
        mean_orig = np.mean(eens_orig)

        # Representative: WEIGHTED mean
        mean_rep = np.sum(prob_rep * eens_rep)

        results['original']['mean'] = mean_orig
        results['representative']['mean'] = mean_rep
        results['errors']['mean_abs'] = abs(mean_rep - mean_orig)
        results['errors']['mean_rel_%'] = abs(mean_rep - mean_orig) / mean_orig * 100

    if 'std' in selected_metrics:
        # Original: unweighted std
        std_orig = np.std(eens_orig, ddof=1)

        # Representative: WEIGHTED std
        mean_rep = np.sum(prob_rep * eens_rep)  # Recalculate if mean wasn't computed
        variance_rep = np.sum(prob_rep * (eens_rep - mean_rep) ** 2)
        std_rep = np.sqrt(variance_rep)

        results['original']['std'] = std_orig
        results['representative']['std'] = std_rep
        results['errors']['std_abs'] = abs(std_rep - std_orig)
        results['errors']['std_rel_%'] = abs(std_rep - std_orig) / std_orig * 100

    if 'variance' in selected_metrics:
        # Original: unweighted variance
        var_orig = np.var(eens_orig, ddof=1)

        # Representative: WEIGHTED variance
        mean_rep = np.sum(prob_rep * eens_rep)
        var_rep = np.sum(prob_rep * (eens_rep - mean_rep) ** 2)

        results['original']['variance'] = var_orig
        results['representative']['variance'] = var_rep
        results['errors']['variance_abs'] = abs(var_rep - var_orig)
        results['errors']['variance_rel_%'] = abs(var_rep - var_orig) / var_orig * 100

    if 'range' in selected_metrics:
        results['original']['min'] = np.min(eens_orig)
        results['original']['max'] = np.max(eens_orig)
        results['original']['range'] = results['original']['max'] - results['original']['min']
        results['representative']['min'] = np.min(eens_rep)
        results['representative']['max'] = np.max(eens_rep)
        results['representative']['range'] = results['representative']['max'] - results['representative']['min']
        results['errors']['range_coverage_%'] = results['representative']['range'] / results['original']['range'] * 100

    if 'percentiles' in selected_metrics:
        percentiles = [10, 25, 50, 75, 90]
        results['original']['percentiles'] = {f'p{p}': np.percentile(eens_orig, p) for p in percentiles}

        # Representative: WEIGHTED percentiles (approximate using sorted cumulative probabilities)
        sorted_indices = np.argsort(eens_rep)
        sorted_eens = eens_rep[sorted_indices]
        sorted_probs = prob_rep[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        rep_percentiles = {}
        for p in percentiles:
            target_prob = p / 100.0
            # Find the EENS value at this cumulative probability
            idx = np.searchsorted(cumulative_probs, target_prob)
            if idx >= len(sorted_eens):
                idx = len(sorted_eens) - 1
            rep_percentiles[f'p{p}'] = sorted_eens[idx]

        results['representative']['percentiles'] = rep_percentiles
        results['errors']['percentiles_rel_%'] = {
            f'p{p}': abs(rep_percentiles[f'p{p}'] - results['original']['percentiles'][f'p{p}'])
                     / results['original']['percentiles'][f'p{p}'] * 100
            for p in percentiles
        }

    # Print report
    if verbose:
        _print_report(results, selected_metrics)

    return results


def _print_report(results: Dict, metrics: List[str]):
    """Print comparison report."""

    info = results['library_info']
    orig = results['original']
    rep = results['representative']
    err = results['errors']

    print("\n" + "=" * 80)
    print("COMPARISON: Original vs Representative Scenarios")
    print("=" * 80)
    print(f"\n  Original:        {info['n_original']} scenarios")
    print(f"  Representative:  {info['n_representative']} scenarios")
    print(f"  Reduction:       {info['reduction_factor']:.1f}x")

    print(f"\n  {'Metric':<20} {'Original':<15} {'Representative':<15} {'Error'}")
    print("  " + "-" * 70)

    if 'mean' in metrics:
        print(f"  {'Mean (GWh)':<20} {orig['mean']:<15.3f} {rep['mean']:<15.3f} {err['mean_rel_%']:>6.2f}%")

    if 'std' in metrics:
        print(f"  {'Std Dev (GWh)':<20} {orig['std']:<15.3f} {rep['std']:<15.3f} {err['std_rel_%']:>6.2f}%")

    if 'variance' in metrics:
        print(
            f"  {'Variance (GWh²)':<20} {orig['variance']:<15.3f} {rep['variance']:<15.3f} {err['variance_rel_%']:>6.2f}%")

    if 'range' in metrics:
        print(f"  {'Min (GWh)':<20} {orig['min']:<15.3f} {rep['min']:<15.3f}")
        print(f"  {'Max (GWh)':<20} {orig['max']:<15.3f} {rep['max']:<15.3f}")
        print(f"  {'Range coverage':<20} {orig['range']:<15.3f} {rep['range']:<15.3f} {err['range_coverage_%']:>6.1f}%")

    if 'percentiles' in metrics:
        print(f"\n  Percentiles:")
        for p in [10, 25, 50, 75, 90]:
            print(f"    {'P' + str(p):<18} {orig['percentiles'][f'p{p}']:<15.3f} "
                  f"{rep['percentiles'][f'p{p}']:<15.3f} {err['percentiles_rel_%'][f'p{p}']:>6.2f}%")

    print("=" * 80 + "\n")


#########################
# SCRIPT ENTRY POINT
#########################

if __name__ == "__main__":
    """
    Example usage demonstrating different convergence scenarios
    """

    # Example 1: Standard convergence-based generation
    print("\n" + "="*80)
    print("EXAMPLE 1: Standard Convergence-Based Generation")
    print("="*80)

    output_path, metrics = generate_windstorm_library_with_convergence(
        network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
        windstorm_preset="windstorm_29_bus_GB_transmission_network",
        convergence_threshold=0.05 ,  # convergence criterion - β
        min_dn_scenarios=100,
        max_dn_scenarios=2000,
        max_generation_attempts=30000,
        initial_batch_size=20,
        base_seed=10000,
        verbose=True,
        visualize_scenarios=True,
        write_lp_on_failure=False,
        write_ilp_on_failure=False,
        output_dir="../Scenario_Database/Scenarios_Libraries/Convergence_Based/",
    )

    # Example 2: Generate representative scenarios using quantile-based PDF splitting
    # print("\n" + "=" * 80)
    # print("EXAMPLE 2: Representative Scenario Selection")
    # print("=" * 80)
    #
    # rep_output_path, selection_info = generate_representative_ws_scenarios(
    #     library_path="../Scenario_Database/Scenarios_Libraries/Convergence_Based/ws_library_29_bus_GB_transmission_network_with_Kearsley_GSP_group_convergence_cov0.020_1834scenarios.json",
    #     output_dir="../Scenario_Database/Scenarios_Libraries/Representatives_from_Convergence_Based/",
    #     save_library=True,
    #     n_representatives=1,
    #     selection_method='equal_interval',
    #     eens_min=None,  # Auto-detect
    #     eens_max=None,  # Auto-detect
    #     handle_empty_intervals='skip',
    #     verbose=True,
    #     visualize_pdf=True,
    #     save_pdf_plot=False,
    # )

    # EXAMPLE 3: Visualize Representative Library PDF
    # print("\n" + "=" * 80)
    # print("EXAMPLE 3: Visualize Representative Library PDF")
    # print("=" * 80)
    #
    # visualize_scenario_library_pdf(
    #     library_path="../Scenario_Database/Scenarios_Libraries/Convergence_Based/ws_library_29_bus_GB_transmission_network_with_Kearsley_GSP_group_convergence_cov0.020_985scenarios.json",
    #     # library_path="../Scenario_Database/Scenarios_Libraries/Representatives_from_Convergence_Based/rep_scn10_from985scn_29BusGB-Kearsley_29GB_s10000_beta0.020.json",
    #     show_kde=True,
    #     kde_bandwidth=0.25,
    #     show_quantile_boundaries=True,  # Show quantile boundaries
    #     show_representatives=True,  # Show representative markers
    #     show_representative_labels=True,  # Show labels
    #     show_mean=False,
    #     show_median=False,
    #     show_cdf=False,
    #     n_bins=40
    # )

    # Example 4: Compare original and representative scenario libraries
    # print("\n" + "=" * 80)
    # print("EXAMPLE 4: Compare All Metrics")
    # print("=" * 80)
    #
    # results_all = compare_original_and_representative_ws_scenarios(
    #     original_library_path="../Scenario_Database/Scenarios_Libraries/Convergence_Based/ws_library_29_bus_GB_transmission_network_with_Kearsley_GSP_group_convergence_cov0.020_1391scenarios.json",
    #     representative_library_path="../Scenario_Database/Scenarios_Libraries/Representatives_from_Convergence_Based/rep_scn8_interval_from1391scn_29BusGB-Kearsley_29GB_seed10000_beta0.020.json",
    #     metrics_to_compare='all'
    # )