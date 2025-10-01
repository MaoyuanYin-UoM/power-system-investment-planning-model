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
from typing import Dict, List, Tuple, Optional
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

def calculate_convergence_metrics(eens_values):
    """
    Calculate convergence metrics based on Billinton & Li (1994).

    Reference:
        R. Billinton and W. Li, "Reliability Assessment of Electric Power
        Systems Using Monte Carlo Methods," Plenum Press, 1994,
        Chapter 3, Section 3.2, Equation (3.7).
    """
    n = len(eens_values)
    mean_eens = np.mean(eens_values)
    std_eens = np.std(eens_values, ddof=1)

    # Raw CoV (for reference)
    cov = std_eens / mean_eens if mean_eens > 0 else float('inf')

    # CONVERGENCE CRITERION: α (called β in our code)
    # α = √V(Q̄)/Q̄ = √V(x)/(Q̄√N) = σ/(μ√n)
    alpha = std_eens / (mean_eens * np.sqrt(n)) if mean_eens > 0 and n > 0 else float('inf')

    return {
        'mean': mean_eens,
        'std': std_eens,
        'cov': cov,  # For information
        'beta': alpha,  # CONVERGENCE CRITERION (Billinton's α)
        'n_scenarios': n
    }


def check_convergence(eens_values, threshold=0.05, min_scenarios=30):
    """
    Check convergence using Billinton & Li (1994) criterion.

    Convergence is achieved when:
        α = σ/(μ√n) < threshold

    where α is the "coefficient of variation" (Eq 3.7) which is actually
    the coefficient of variation of the MEAN estimate (β or relative SE).

    Reference:
        R. Billinton and W. Li, "Reliability Assessment of Electric Power
        Systems Using Monte Carlo Methods," Plenum Press, 1994,
        Chapter 3, Equations (3.7) and (3.8), page 36-38.
    """
    metrics = calculate_convergence_metrics(eens_values)

    if metrics['n_scenarios'] < min_scenarios:
        return False, metrics

    # Check α criterion (we call it beta in our code)
    converged = metrics['beta'] < threshold

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

    print("\n" + "="*60)
    print("PROGRESS REPORT")
    print("="*60)
    print(f"Total scenarios generated: {n_total_generated}")
    print(f"DN-affecting scenarios: {n_dn_scenarios} ({hit_rate:.1%} hit rate)")

    if metrics['n_scenarios'] > 0:
        print(f"\nConvergence Metrics:")
        print(f"  Mean EENS (DN): {metrics['mean']:.3f} GWh")
        print(f"  Std Dev EENS: {metrics['std']:.3f} GWh")
        print(f"  CoV: {metrics['cov']:.4f}")
        print(f"  95% CI: [{metrics['ci_lower']:.3f}, {metrics['ci_upper']:.3f}] GWh")

    print(f"\nElapsed time: {elapsed_time:.1f} seconds")
    print(f"Generation rate: {n_total_generated/elapsed_time:.1f} scenarios/sec")
    print("="*60)


#########################
# MAIN GENERATION FUNCTION
#########################

def generate_windstorm_library_with_convergence(
        network_preset: str,
        windstorm_preset: str,
        convergence_threshold: float = 0.05,
        min_dn_scenarios: int = 30,
        max_dn_scenarios: int = 100,
        max_generation_attempts: int = 500,
        initial_batch_size: int = 10,
        base_seed: int = 42,
        output_dir: str = "../Scenario_Database/Scenarios_Libraries/",
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
        convergence_threshold: CoV threshold for convergence (default 0.05 = 5%)
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
                dn_scenarios[scenario_id]['eens_dn'] = eens_dn
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
                "method": "coefficient_of_variation",
                "threshold": convergence_threshold,
                "converged": converged,
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


def generate_representative_ws_scenarios_by_splitting_pdf(
        library_path: str,
        n_representatives: int = 5,
        output_dir: str = "../Scenario_Database/Scenarios_Libraries/Representative/",
        library_name: Optional[str] = None,
        total_windstorm_probability: float = 0.01,
        verbose: bool = True
) -> Tuple[str, Dict]:
    """
    Generate representative windstorm scenarios using quantile-based PDF splitting.

    This function:
    1. Loads a convergence-based scenario library with EENS values
    2. Sorts scenarios by EENS to construct empirical PDF
    3. Divides into equal-probability quantiles
    4. Selects middle scenario from each quantile as representative
    5. Saves as new library with adjusted probabilities

    Args:
        library_path: Path to convergence-based scenario library JSON
        n_representatives: Number of representative scenarios to select
        output_dir: Directory to save representative library
        library_name: Output filename (auto-generated if None)
        total_windstorm_probability: Total probability to distribute among representatives
        verbose: Print progress and statistics

    Returns:
        (output_path, selection_info): Path to saved library and selection statistics
    """

    if verbose:
        print("\n" + "=" * 70)
        print("REPRESENTATIVE SCENARIO SELECTION (Quantile-Based)")
        print("=" * 70)
        print(f"Source library: {library_path}")
        print(f"Target representatives: {n_representatives}")

    # ====================
    # 1. LOAD SOURCE LIBRARY
    # ====================

    with open(library_path, 'r') as f:
        full_library = json.load(f)

    scenarios = full_library['scenarios']
    source_metadata = full_library['metadata']

    if verbose:
        print(f"Loaded {len(scenarios)} scenarios from library")

    # ====================
    # 2. SORT BY EENS
    # ====================

    # Create list of (scenario_id, eens_value, full_scenario_data)
    scenario_list = []
    for sid, sdata in scenarios.items():
        if 'eens_dn_gwh' not in sdata:
            raise ValueError(f"Scenario {sid} missing 'eens_dn_gwh' field")
        scenario_list.append((sid, sdata['eens_dn_gwh'], sdata))

    # Sort by EENS value (ascending)
    scenario_list.sort(key=lambda x: x[1])

    eens_values = [x[1] for x in scenario_list]
    min_eens = min(eens_values)
    max_eens = max(eens_values)
    mean_eens = np.mean(eens_values)

    if verbose:
        print(f"\nEENS Statistics:")
        print(f"  Range: {min_eens:.3f} - {max_eens:.3f} GWh")
        print(f"  Mean: {mean_eens:.3f} GWh")
        print(f"  Median: {np.median(eens_values):.3f} GWh")

    # ====================
    # 3. QUANTILE-BASED SELECTION
    # ====================

    n_scenarios = len(scenario_list)

    # Calculate quantile boundaries
    quantile_size = n_scenarios / n_representatives

    representatives = {}
    selection_info = {
        'quantiles': [],
        'selected_scenarios': []
    }

    if verbose:
        print(f"\nSelecting representatives from {n_representatives} quantiles:")
        print(f"  Each quantile contains ~{quantile_size:.1f} scenarios")
        print("\nQuantile Selection:")

    for i in range(n_representatives):
        # Define quantile boundaries (using floating point for better distribution)
        start_idx = int(i * quantile_size)
        end_idx = int((i + 1) * quantile_size) if i < n_representatives - 1 else n_scenarios

        # Select middle scenario from this quantile
        middle_idx = (start_idx + end_idx) // 2

        # Get the selected scenario
        scenario_id, eens_value, scenario_data = scenario_list[middle_idx]

        # Create representative ID
        rep_id = f"rep_{i:02d}"

        # Calculate quantile statistics
        quantile_eens = [x[1] for x in scenario_list[start_idx:end_idx]]
        quantile_min = min(quantile_eens)
        quantile_max = max(quantile_eens)
        quantile_mean = np.mean(quantile_eens)

        # Store representative
        representatives[rep_id] = {
            'original_id': scenario_id,
            'scenario_data': scenario_data,
            'eens_gwh': eens_value,
            'probability': 1.0 / n_representatives,  # Equal probability for each quantile
            'quantile_info': {
                'quantile_number': i + 1,
                'quantile_range': f"{(i / n_representatives) * 100:.0f}-{((i + 1) / n_representatives) * 100:.0f}%",
                'scenarios_in_quantile': end_idx - start_idx,
                'eens_range_gwh': [quantile_min, quantile_max],
                'eens_mean_gwh': quantile_mean,
                'position_in_quantile': 'middle'
            }
        }

        # Store selection info
        selection_info['quantiles'].append({
            'quantile': i + 1,
            'selected_scenario': scenario_id,
            'selected_eens': eens_value,
            'quantile_size': end_idx - start_idx
        })

        if verbose:
            print(f"  Q{i + 1} ({quantile_min:.1f}-{quantile_max:.1f} GWh): "
                  f"Selected {scenario_id} with EENS = {eens_value:.2f} GWh")

    # ====================
    # 4. CREATE REPRESENTATIVE LIBRARY
    # ====================

    # Calculate expected EENS from representatives
    expected_eens = sum(
        rep['eens_gwh'] * rep['probability']
        for rep in representatives.values()
    )

    representative_library = {
        "metadata": {
            "library_type": "representative_windstorm_scenarios",
            "network_preset": source_metadata.get('network_preset'),
            "windstorm_preset": source_metadata.get('windstorm_preset'),
            "num_scenarios": n_representatives,
            "generation_date": datetime.now().isoformat(),
            "scenario_id_format": "rep_XX (2-digit zero-padded)",

            "source_library": {
                "path": library_path,
                "num_scenarios": len(scenarios),
                "eens_range_gwh": [min_eens, max_eens],
                "convergence_info": source_metadata.get('convergence_info', {})
            },

            "selection_method": {
                "method": "quantile_based",
                "description": "Equal-probability quantiles with middle scenario selection",
                "n_representatives": n_representatives,
                "probability_per_representative": 1.0 / n_representatives,
                "total_windstorm_probability": total_windstorm_probability
            },

            "representative_statistics": {
                "eens_values_gwh": [rep['eens_gwh'] for rep in representatives.values()],
                "probabilities": [rep['probability'] for rep in representatives.values()],
                "expected_eens_gwh": expected_eens,
                "coverage": {
                    "min_eens_gwh": min([rep['eens_gwh'] for rep in representatives.values()]),
                    "max_eens_gwh": max([rep['eens_gwh'] for rep in representatives.values()]),
                    "captures_range_percent": (
                        (max([rep['eens_gwh'] for rep in representatives.values()]) -
                         min([rep['eens_gwh'] for rep in representatives.values()])) /
                        (max_eens - min_eens) * 100 if max_eens > min_eens else 100
                    )
                }
            }
        },

        "scenarios": {}
    }

    # Add scenarios with adjusted probabilities
    for rep_id, rep_info in representatives.items():
        # Copy full scenario data
        scenario_data = rep_info['scenario_data'].copy()

        # Add probability for optimization (share of total windstorm probability)
        scenario_data['probability'] = rep_info['probability'] * total_windstorm_probability

        # Add metadata about selection
        scenario_data['representative_metadata'] = {
            'original_scenario_id': rep_info['original_id'],
            'quantile_info': rep_info['quantile_info'],
            'selection_method': 'quantile_middle',
            'eens_gwh': rep_info['eens_gwh']
        }

        # Use representative ID
        representative_library['scenarios'][rep_id] = scenario_data

    # ====================
    # 5. SAVE LIBRARY
    # ====================

    # Generate output filename if not provided
    if library_name is None:
        # Extract network name from source if possible
        network_name = source_metadata.get('network_preset', 'network').replace('/', '_')
        library_name = f"representative_quantile_{n_representatives}scenarios_{network_name}.json"

    output_path = os.path.join(output_dir, library_name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save library
    with open(output_path, 'w') as f:
        json.dump(representative_library, f, indent=2, default=str)

    # Get file size
    file_size_kb = os.path.getsize(output_path) / 1024

    # ====================
    # 6. FINAL REPORT
    # ====================

    if verbose:
        print("\n" + "=" * 70)
        print("SELECTION COMPLETE")
        print("=" * 70)
        print(f"Representatives selected: {n_representatives}")
        print(f"Expected EENS: {expected_eens:.3f} GWh")
        print(f"Each representative probability: {1.0 / n_representatives:.3f}")
        print(f"Total windstorm probability: {total_windstorm_probability}")
        print(f"\nOutput file: {output_path}")
        print(f"File size: {file_size_kb:.1f} KB")

        print("\nRepresentative Summary:")
        for i, (rep_id, rep) in enumerate(representatives.items()):
            print(f"  {rep_id}: EENS = {rep['eens_gwh']:6.2f} GWh, "
                  f"P = {rep['probability']:.3f}, "
                  f"From Q{i + 1} (orig: {rep['original_id']})")

    # Prepare return statistics
    selection_statistics = {
        'n_source_scenarios': len(scenarios),
        'n_representatives': n_representatives,
        'expected_eens_gwh': expected_eens,
        'eens_range': [min_eens, max_eens],
        'representative_eens': [rep['eens_gwh'] for rep in representatives.values()],
        'output_path': output_path,
        'selection_details': selection_info
    }

    return output_path, selection_statistics


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
        convergence_threshold=0.05,  # Coefficient of Variance (CoV)
        min_dn_scenarios=40,
        max_dn_scenarios=1000,
        initial_batch_size=20,
        base_seed=10000,
        verbose=True,
        visualize_scenarios=True,
        write_lp_on_failure=False,
        write_ilp_on_failure=False,
    )

    # Example 2: Stricter convergence for higher confidence
    # print("\n" + "="*80)
    # print("EXAMPLE 2: Stricter Convergence Criteria")
    # print("="*80)
    #
    # output_path, metrics = generate_windstorm_library_with_convergence(
    #     network_preset="tn29_dn38_kearsley_gsp",
    #     windstorm_preset="ctr_model",
    #     convergence_threshold=0.02,  # 2% CoV - very strict
    #     min_dn_scenarios=50,
    #     max_dn_scenarios=200,
    #     max_generation_attempts=5000,
    #     base_seed=20000,
    #     verbose=True
    # )

    # Example 3: Quick generation for testing
    # print("\n" + "="*80)
    # print("EXAMPLE 3: Quick Test Generation")
    # print("="*80)
    #
    # output_path, metrics = generate_windstorm_library_with_convergence(
    #     network_preset="tn29_dn38_kearsley_gsp",
    #     windstorm_preset="ctr_model",
    #     convergence_threshold=0.10,  # 10% CoV - loose
    #     min_dn_scenarios=10,
    #     max_dn_scenarios=20,
    #     initial_batch_size=20,
    #     base_seed=30000,
    #     verbose=True
    # )