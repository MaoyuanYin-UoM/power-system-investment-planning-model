"""
Windstorm Scenario Library Generator

Generates a library of windstorm scenarios for use in multi-stage stochastic optimization.
Each scenario is given a unique ID and stored in a single JSON file.
"""

import os
import json
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Tuple

from factories.windstorm_factory import make_windstorm
from factories.network_factory import make_network


def generate_windstorm_library(
        network_preset: str,
        windstorm_preset: str,
        num_scenarios: int = 100,
        base_seed: int = 10000,
        output_dir: str = "../Scenario_Database/Scenarios_for_Scenario_Tree/Original_Scenario_Libraries",
        library_name: Optional[str] = None
) -> str:
    """
    Generate a library of windstorm scenarios with unique IDs.
    Each scenario is generated independently with its own seed (scenario_seed = base_seed + scenario_idx).
    """
    print(f"Generating windstorm scenario library...")
    print(f"  Network: {network_preset}")
    print(f"  Windstorm preset: {windstorm_preset}")
    print(f"  Number of scenarios: {num_scenarios}")
    print(f"  Base seed: {base_seed}")
    print(f"  Output Directory: {output_dir}")

    # Load network (this is shared across all scenarios)
    net = make_network(network_preset)

    # Get network parameters
    num_bus = len(net.data.net.bus)
    num_bch = len(net.data.net.bch)

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

    # Initialize library structure
    library = {
        "metadata": {
            "library_type": "windstorm_scenarios",
            "network_preset": network_preset,
            "windstorm_preset": windstorm_preset,
            "num_scenarios": num_scenarios,
            "base_seed": base_seed,
            "generation_date": datetime.now().isoformat(),
            "scenario_id_format": "ws_XXXX (4-digit zero-padded)",
            "network_info": {
                "num_buses": num_bus,
                "num_branches": num_bch
            }
        },
        "scenarios": {}
    }

    # Generate scenarios
    print("\nGenerating scenarios...")

    for scenario_idx in range(num_scenarios):
        scenario_id = f"ws_{scenario_idx:04d}"
        print(f"\rGenerating scenario {scenario_idx + 1}/{num_scenarios} (ID: {scenario_id})...", end="")

        # Create a unique seed for this scenario
        scenario_seed = base_seed + scenario_idx
        np.random.seed(scenario_seed)

        # Create a fresh WindClass instance for this scenario
        ws = make_windstorm(windstorm_preset)

        # Initialize contour distances
        ws.init_ws_path0()

        # Get period info
        num_hrs_prd = ws._get_num_hrs_prd()

        # Initialize scenario data
        scenario_data = {
            "scenario_id": scenario_id,
            "scenario_index": scenario_idx,
            "scenario_seed": scenario_seed,  # Store the unique seed
            "events": [],
        }

        # Generate number of events for this scenario
        num_events = ws.MC.WS.num_ws_prd[0]

        # Generate repair times for all branches
        # (It is assumed that for each branch, the 'ttr' is shared across all events in a scenario)
        ttr_min, ttr_max = ws.data.WS.event.ttr
        bch_ttr = np.random.randint(ttr_min, ttr_max + 1, size=num_bch)

        # Generate start/end locations for all events in this scenario
        start_lons, start_lats, end_lons, end_lats = ws.init_ws_path(num_events)

        # Generate timing for events
        max_lng = max(ws.data.WS.event.lng) + max(ws.data.WS.event.ttr)
        event_start_times = []

        for event_idx in range(num_events):
            lng_ws = ws.MC.WS.lng[event_idx]

            if event_idx == 0:
                # First event can start anywhere in the year (as long as the event ends before the end of the year)
                max_start = num_hrs_prd - lng_ws - max(bch_ttr)
                ts = np.random.randint(1, max_start + 1)
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
        for event_idx in range(num_events):
            ts = event_start_times[event_idx]

            # Get duration for this specific event
            lng_ws = ws.MC.WS.lng[event_idx]

            # Get wind speed limits for this event
            lim_v_ws = ws.MC.WS.lim_v_ws_all[event_idx]

            # Create windstorm path
            path_ws = ws.crt_ws_path(
                start_lons[event_idx], start_lats[event_idx],
                end_lons[event_idx], end_lats[event_idx],
                lng_ws
            )

            # Create windstorm radius over time
            radius_ws = ws.crt_ws_radius(lng_ws)

            # Create wind speed (single value for entire event)
            v_ws = ws.crt_ws_v(lim_v_ws, lng_ws)

            # Generate random numbers ONLY for event duration
            bch_rand_nums = np.random.rand(num_bch, lng_ws)

            # Calculate branch impacts ONLY during windstorm duration
            flgs_impacted_bch = np.zeros((num_bch, lng_ws), dtype=int)

            for t in range(lng_ws):
                epicentre = path_ws[t]
                flgs_impacted_bch[:, t] = np.array(
                    ws.compare_circle(epicentre, radius_ws[t], bch_gis_bgn, bch_gis_end, num_bch),
                    dtype=int
                )

                # Store all event data in one place
            event_data = {
                "event_id": int(actual_event_idx + 1),
                "bgn_hr": int(ts),  # Random hour in [1, 8760]
                "duration": int(lng_ws),
                "epicentre": [list(point) for point in path_ws[:lng_ws]],  # Hourly positions
                "radius": radius_ws[:lng_ws].tolist(),  # Hourly radius values
                "gust_speed": v_ws,  # Hourly wind speeds (linear decay)
                "bch_rand_nums": bch_rand_nums.tolist(),  # [num_branches x duration]
                "flgs_impacted_bch": flgs_impacted_bch.tolist(),  # [num_branches x duration]
                "bch_ttr": [int(x) for x in bch_ttr]  # Repair times for each branch
            }
            scenario_data["events"].append(event_data)
            actual_event_idx += 1

            # Add to library
        library["scenarios"][scenario_id] = scenario_data

        print(f"\n\n✓ Generated {num_scenarios} scenarios successfully")

    # Save library
    os.makedirs(output_dir, exist_ok=True)

    # Create filename
    if library_name is None:
        network_alias = _get_network_alias(network_preset)
        windstorm_alias = _get_windstorm_alias(windstorm_preset)
        library_name = f"windstorm_library_{network_alias}_{windstorm_alias}_{num_scenarios}scenarios_seed{base_seed}.json"

    output_path = os.path.join(output_dir, library_name)

    with open(output_path, 'w') as f:
        json.dump(library, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Windstorm scenario library saved to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Average size per scenario: {file_size_mb / num_scenarios * 1000:.2f} KB")

    return output_path


def _get_network_alias(network_preset: str) -> str:
    """Get shortened network alias for filename."""
    aliases = {
        "29_bus_GB_transmission_network_with_Kearsley_GSP_group": "29BusGB-KearsleyGSP",
        # Add more aliases as needed
    }
    return aliases.get(network_preset, network_preset)


def _get_windstorm_alias(windstorm_preset: str) -> str:
    """Get shortened windstorm alias for filename."""
    aliases = {
        "windstorm_GB_transmission_network": "GB",
        # Add more aliases as needed
    }
    return aliases.get(windstorm_preset, windstorm_preset)


def load_scenario_from_library(library_path: str, scenario_id: str) -> Dict:
    """
    Load a specific scenario from the library.

    Args:
        library_path: Path to the library JSON file
        scenario_id: ID of the scenario to load (e.g., "ws_0042")

    Returns:
        Scenario data dictionary
    """
    with open(library_path, 'r') as f:
        library = json.load(f)

    if scenario_id not in library["scenarios"]:
        raise ValueError(f"Scenario {scenario_id} not found in library")

    return library["scenarios"][scenario_id]


def load_multiple_scenarios(library_path: str, scenario_ids: List[str]) -> Dict[str, Dict]:
    """
    Load multiple scenarios from the library at once.

    Args:
        library_path: Path to the library JSON file
        scenario_ids: List of scenario IDs to load

    Returns:
        Dictionary mapping scenario IDs to their data
    """
    with open(library_path, 'r') as f:
        library = json.load(f)

    scenarios = {}
    for scenario_id in scenario_ids:
        if scenario_id not in library["scenarios"]:
            raise ValueError(f"Scenario {scenario_id} not found in library")
        scenarios[scenario_id] = library["scenarios"][scenario_id]

    return scenarios


def get_library_metadata(library_path: str) -> Dict:
    """Get metadata about the scenario library."""
    with open(library_path, 'r') as f:
        library = json.load(f)
    return library["metadata"]


def list_scenario_ids(library_path: str) -> List[str]:
    """List all scenario IDs in the library."""
    with open(library_path, 'r') as f:
        library = json.load(f)
    return sorted(list(library["scenarios"].keys()))


def sample_scenarios_from_library(library_path: str, num_samples: int,
                                seed: Optional[int] = None) -> List[str]:
    """
    Randomly sample scenario IDs from the library.

    Note that in the future this might be replaced by more justified approaches (e.g., Clustering methods)

    Args:
        library_path: Path to the library JSON file
        num_samples: Number of scenarios to sample
        seed: Seed for sampling scenarios. This arg is adopted for reproducibility

    Returns:
        List of sampled scenario IDs
    """
    if seed is not None:
        np.random.seed(seed)

    scenario_ids = list_scenario_ids(library_path)

    if num_samples > len(scenario_ids):
        raise ValueError(f"Cannot sample {num_samples} scenarios from library with only {len(scenario_ids)} scenarios")

    return np.random.choice(scenario_ids, size=num_samples, replace=False).tolist()


def expand_windstorm_windows_to_full_period(scenario_data: Dict, num_branches: int,
                                           num_hours: int = 8760) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand windstorm window data back to full period arrays if needed.

    Args:
        scenario_data: Scenario data with windstorm windows
        num_branches: Number of branches in the network
        num_hours: Total hours in the period (default 8760)

    Returns:
        Tuple of (bch_rand_nums, flgs_impacted_bch) as full period arrays
    """
    # Initialize full arrays
    bch_rand_nums = np.random.rand(num_branches, num_hours)  # Default random values
    flgs_impacted_bch = np.zeros((num_branches, num_hours), dtype=int)

    # Fill in windstorm window data
    for window in scenario_data.get("windstorm_windows", []):
        start = window["window_start_hr"]
        end = window["window_end_hr"]

        # Insert window data into full arrays
        bch_rand_nums[:, start:end] = np.array(window["bch_rand_nums"])
        flgs_impacted_bch[:, start:end] = np.array(window["flgs_impacted_bch"])

    return bch_rand_nums, flgs_impacted_bch


def get_scenario_statistics(library_path: str) -> Dict:
    """
    Get statistics about scenarios in the library.

    Returns dictionary with statistics like:
    - Average number of events per scenario
    - Average event duration
    - Total windstorm window hours
    - Storage efficiency metrics
    """
    with open(library_path, 'r') as f:
        library = json.load(f)

    stats = {
        "num_scenarios": len(library["scenarios"]),
        "total_events": 0,
        "total_event_duration": 0,
        "total_window_hours": 0,
        "max_events_per_scenario": 0,
        "min_events_per_scenario": float('inf'),
        "avg_repair_time": 0
    }

    total_repair_times = []

    for scenario_id, scenario in library["scenarios"].items():
        num_events = len(scenario["events"])
        stats["total_events"] += num_events
        stats["max_events_per_scenario"] = max(stats["max_events_per_scenario"], num_events)
        stats["min_events_per_scenario"] = min(stats["min_events_per_scenario"], num_events)

        for event in scenario["events"]:
            stats["total_event_duration"] += event["duration"]

        # Calculate total window hours
        for window in scenario.get("windstorm_windows", []):
            stats["total_window_hours"] += window["window_duration"]
            total_repair_times.extend(window["bch_ttr"])

    stats["avg_events_per_scenario"] = stats["total_events"] / stats["num_scenarios"]
    stats["avg_event_duration"] = stats["total_event_duration"] / stats["total_events"] if stats["total_events"] > 0 else 0
    stats["avg_window_hours"] = stats["total_window_hours"] / stats["num_scenarios"]
    stats["avg_repair_time"] = np.mean(total_repair_times) if total_repair_times else 0

    # Calculate storage efficiency
    metadata = library.get("metadata", {})
    if "network_info" in metadata and "windstorm_info" in metadata:
        num_branches = metadata["network_info"]["num_branches"]
        total_hours = metadata["windstorm_info"]["num_hours"]
        full_storage_hours = num_branches * total_hours * stats["num_scenarios"]
        actual_storage_hours = stats["total_window_hours"] * num_branches
        stats["storage_efficiency"] = f"{(1 - actual_storage_hours/full_storage_hours)*100:.1f}%"

    return stats


if __name__ == "__main__":
    # Example usage
    library_path = generate_windstorm_library(
        network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
        windstorm_preset="windstorm_GB_transmission_network",
        num_scenarios=100,
        base_seed=10000,
        output_dir="../Scenario_Database/Scenarios_for_Scenario_Tree/Original_Scenario_Libraries",
    )

    # Test loading functionality
    print("\n--- Testing Library Functions ---")

    # Get metadata
    metadata = get_library_metadata(library_path)
    print(f"\nLibrary metadata:")
    print(f"  Scenarios: {metadata['num_scenarios']}")
    print(f"  Base seed: {metadata['base_seed']}")
    print(f"  Network buses: {metadata['network_info']['num_buses']}")
    print(f"  Network branches: {metadata['network_info']['num_branches']}")

    # Get statistics
    stats = get_scenario_statistics(library_path)
    print(f"\nScenario statistics:")
    print(f"  Average events per scenario: {stats['avg_events_per_scenario']:.2f}")
    print(f"  Average event duration: {stats['avg_event_duration']:.2f} hours")
    print(f"  Average window hours per scenario: {stats['avg_window_hours']:.2f} hours")
    print(f"  Average repair time: {stats['avg_repair_time']:.2f} hours")
    if 'storage_efficiency' in stats:
        print(f"  Storage efficiency: {stats['storage_efficiency']} saved vs full 8760-hour storage")

    # List first 5 scenario IDs
    scenario_ids = list_scenario_ids(library_path)
    print(f"\nFirst 5 scenario IDs: {scenario_ids[:5]}")

    # Load a specific scenario
    if scenario_ids:
        first_scenario = load_scenario_from_library(library_path, scenario_ids[0])
        print(f"\nLoaded scenario {scenario_ids[0]}:")
        print(f"  Number of events: {len(first_scenario.get('events', []))}")
        if first_scenario.get('events'):
            print(f"  First event start hour: {first_scenario['events'][0]['bgn_hr']}")
            print(f"  First event duration: {first_scenario['events'][0]['duration']} hours")