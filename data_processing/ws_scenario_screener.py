"""
Windstorm Scenario Screener

Filters windstorm scenarios based on:
1. Spatial proximity to distribution network
2. Impact on distribution network branches
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

# Import visualization functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization import visualize_windstorm_event


def load_library(library_path: str) -> Dict:
    """Load windstorm scenario library."""
    with open(library_path, 'r') as f:
        return json.load(f)


def save_filtered_library(library: Dict, output_path: str):
    """Save filtered library to JSON."""
    with open(output_path, 'w') as f:
        json.dump(library, f, indent=2)
    
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✓ Filtered library saved to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")


def extract_library_info(library: Dict) -> Tuple[str, str, int, int]:
    """
    Extract information directly from library metadata.
    
    Returns:
        Tuple of (network_preset, windstorm_preset, num_scenarios, base_seed)
    """
    metadata = library.get("metadata", {})
    
    network_preset = metadata.get("network_preset", "unknown_network")
    windstorm_preset = metadata.get("windstorm_preset", "unknown_windstorm")
    num_scenarios = metadata.get("num_scenarios", 0)
    base_seed = metadata.get("base_seed", 0)
    
    return network_preset, windstorm_preset, num_scenarios, base_seed


def get_alias(preset_name: str, preset_type: str) -> str:
    """
    Get alias for network or windstorm preset names.
    
    Args:
        preset_name: The full preset name
        preset_type: Either "network" or "windstorm"
    
    Returns:
        Alias if defined, otherwise the full preset name
    """
    if preset_type == "network":
        if preset_name == "29_bus_GB_transmission_network_with_Kearsley_GSP_group":
            return "29BusGB-KearsleyGSP"
        else:
            return preset_name
    elif preset_type == "windstorm":
        if preset_name == "windstorm_GB_transmission_network":
            return "GB"
        else:
            return preset_name
    else:
        return preset_name


def generate_output_filename(library: Dict,
                           min_impacted_branches: int,
                           min_impact_hours: int,
                           buffer_km: float) -> str:
    """
    Generate output filename based on library metadata and screening parameters.
    
    Format: windstorm_library_{network}_{ws}_{N}scn_s{S}_filt_b{B}_h{H}_buf{BUF}.json
    """
    # Extract info from metadata
    network_preset, windstorm_preset, num_scenarios, base_seed = extract_library_info(library)
    
    # Get aliases
    network_alias = get_alias(network_preset, "network")
    ws_alias = get_alias(windstorm_preset, "windstorm")
    
    # Build filename
    output_name = (
        f"ws_library_{network_alias}_{ws_alias}_"
        f"{num_scenarios}scn_s{base_seed}_"
        f"filt_b{min_impacted_branches}_h{min_impact_hours}_buf{int(buffer_km)}.json"
    )
    
    return output_name


def get_distribution_branch_indices(network_preset: str, 
                                  custom_indices: Optional[List[int]] = None) -> List[int]:
    """
    Get indices of distribution network branches for a given network.
    
    Args:
        network_preset: Network preset name
        custom_indices: Optional custom list of distribution branch indices
    
    Returns:
        List of distribution branch indices
    """
    if custom_indices is not None:
        return custom_indices

    if network_preset == "29_bus_GB_transmission_network_with_Kearsley_GSP_group":
        return list(range(30, 37)) + list(range(46, 77))
    else:
        print(f"Warning: No distribution branch indices defined for network '{network_preset}'")
        return []


def define_distribution_network_boundary(network_preset: str, 
                          custom_boundary: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Define geographical boundary for a given network's distribution area.
    
    Args:
        network_preset: Network preset name
        custom_boundary: Optional custom boundary dict with keys: min_lat, max_lat, min_lon, max_lon
    
    Returns:
        Dictionary with boundary coordinates
    """
    if custom_boundary is not None:
        return custom_boundary

    if network_preset == "29_bus_GB_transmission_network_with_Kearsley_GSP_group":
        from factories.network_factory import make_network

        # Load the distribution network
        dn_network = make_network("Manchester_distribution_network_Kearsley")

        # Get all bus coordinates
        bus_lats = dn_network.data.net.bus_lat
        bus_lons = dn_network.data.net.bus_lon

        # Calculate boundaries from actual bus locations
        boundary = {
            "min_lat": min(bus_lats),
            "max_lat": max(bus_lats),
            "min_lon": min(bus_lons),
            "max_lon": max(bus_lons)
        }
        return boundary

    else:
        print(f"Warning: No boundary defined for network '{network_preset}', using default")
        return {
            "min_lat": 53,
            "max_lat": 54,
            "min_lon": -3,
            "max_lon": -2
        }


def point_in_expanded_boundary(lat: float, lon: float, boundary: Dict[str, float], 
                               buffer_km: float = 50) -> bool:
    """
    Check if point is within expanded boundary (including buffer).
    
    Simple approximation: 1 degree ≈ 111 km at this latitude
    """
    buffer_deg = buffer_km / 111.0
    
    return (boundary["min_lat"] - buffer_deg <= lat <= boundary["max_lat"] + buffer_deg and
            boundary["min_lon"] - buffer_deg <= lon <= boundary["max_lon"] + buffer_deg)


def passes_spatial_filter(scenario: Dict, boundary: Dict[str, float],
                          buffer_km: float = 50, min_impact_hours: int = 1) -> bool:
    """
    Check if scenario passes spatial filtering.

    Returns True if the total hours with epicenters within the DN boundary >= min_impact_hours.
    """
    total_hours_in_boundary = 0

    for event in scenario.get("events", []):
        # Check each epicenter position (each represents one hour)
        for epicenter in event.get("epicentre", []):
            if point_in_expanded_boundary(epicenter[1], epicenter[0], boundary, buffer_km):
                total_hours_in_boundary += 1

    return total_hours_in_boundary >= min_impact_hours


def passes_impact_filter(scenario: Dict, dn_branch_indices: List[int],
                         min_impacted_branches: int = 1) -> bool:
    """
    Check if scenario impacts distribution network branches.

    Args:
        scenario: Scenario data
        dn_branch_indices: Indices of distribution branches
        min_impacted_branches: Minimum number of unique DN branches that must be impacted

    Returns True if sufficient unique DN branches are impacted (at least one hour).
    """
    # Use a set to track unique impacted branches
    impacted_dn_branches = set()

    for event in scenario.get("events", []):
        # Get impact flags directly from event (new structure)
        impact_flags = np.array(event.get("flgs_impacted_bch", []))

        if len(impact_flags) == 0:
            continue

        # Check each distribution branch
        for dn_idx in dn_branch_indices:
            if dn_idx < len(impact_flags):
                # Check if this branch is impacted at any hour during this event
                if np.any(impact_flags[dn_idx] > 0):
                    impacted_dn_branches.add(dn_idx)

    return len(impacted_dn_branches) >= min_impacted_branches


def screen_scenarios(library_path: str = "../Scenario_Database/Scenarios_Libraries/Original_Scenario_Libraries",
                     output_dir: Optional[
                         str] = "../Scenario_Database/Scenarios_Libraries/Filtered_Scenario_Libraries",
                     # Change parameter name
                     buffer_km: float = 50,
                     min_impacted_branches: int = 1,
                     min_impact_hours: int = 1,
                     custom_dn_indices: Optional[List[int]] = None,
                     custom_boundary: Optional[Dict[str, float]] = None) -> Dict:
    """
    Main screening function.

    Args:
        library_path: Path to input library JSON
        output_dir: Directory to save filtered library (uses default if None)  # Updated description
        buffer_km: Buffer around DN boundary for spatial filter
        min_impacted_branches: Minimum DN (failable) branches that must be impacted to pass the screening
        min_impact_hours: Minimum number of hours that the windstorm event epicentre must fall within the DN boundary
                          (the exact boundary plus a buffer distance) to pass the screening
        custom_dn_indices: Optional custom DN (failable) branch indices
        custom_boundary: Optional custom DN boundary lons and lats

    Returns:
        Dictionary containing filtered library path and statistics
    """
    print("Loading windstorm scenario library...")
    library = load_library(library_path)
    
    # Get network preset from metadata
    network_preset = library["metadata"].get("network_preset", "")
    
    # Get distribution branch indices and boundary
    dn_branch_indices = get_distribution_branch_indices(network_preset, custom_dn_indices)
    boundary = define_distribution_network_boundary(network_preset, custom_boundary)
    
    print(f"\nNetwork: {network_preset}")
    print(f"\nScreening parameters:")
    print(f"  Distribution branches: {len(dn_branch_indices)} branches")
    print(f"  Spatial buffer: {buffer_km} km")
    print(f"  Min impacted branches: {min_impacted_branches}")
    print(f"  Min impact hours: {min_impact_hours}")
    print(f"  Boundary: Lat [{boundary['min_lat']:.2f}, {boundary['max_lat']:.2f}], "
          f"Lon [{boundary['min_lon']:.2f}, {boundary['max_lon']:.2f}]")
    
    # Statistics
    stats = {
        "total_scenarios": len(library["scenarios"]),
        "passed_spatial": 0,
        "passed_impact": 0,
        "passed_both": 0,
        "filtered_scenario_ids": []
    }
    
    # Filter scenarios
    filtered_scenarios = {}

    print("\nScreening scenarios...")
    for scenario_id, scenario in library["scenarios"].items():
        # Check both filters independently
        spatial_pass = passes_spatial_filter(scenario, boundary, buffer_km, min_impact_hours)
        impact_pass = passes_impact_filter(scenario, dn_branch_indices, min_impacted_branches)

        # Update statistics
        if spatial_pass:
            stats["passed_spatial"] += 1
        if impact_pass:
            stats["passed_impact"] += 1

        # Check if both filters passed
        if spatial_pass and impact_pass:
            stats["passed_both"] += 1
            filtered_scenarios[scenario_id] = scenario
            stats["filtered_scenario_ids"].append(scenario_id)
    
    # Create filtered library
    filtered_library = {
        "metadata": library["metadata"].copy(),
        "scenarios": filtered_scenarios
    }
    
    # Update metadata
    filtered_library["metadata"]["screening_info"] = {
        "original_scenarios": stats["total_scenarios"],
        "filtered_scenarios": stats["passed_both"],
        "buffer_km": buffer_km,
        "min_impacted_branches": min_impacted_branches,
        "min_impact_hours": min_impact_hours,
        "screening_date": datetime.now().isoformat()
    }

    # Generate output path
    output_filename = generate_output_filename(
        library, min_impacted_branches, min_impact_hours, buffer_km
    )

    # Ensure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)  # Now this works correctly
    else:
        output_path = output_filename  # Just use filename if no directory specified

    # Save filtered library
    save_filtered_library(filtered_library, output_path)
    
    # Print statistics
    print(f"\nScreening Results:")
    print(f"  Original scenarios: {stats['total_scenarios']}")
    print(f"  Passed spatial filter: {stats['passed_spatial']} "
          f"({stats['passed_spatial']/stats['total_scenarios']*100:.1f}%)")
    print(f"  Passed impact filter: {stats['passed_impact']} "
          f"({stats['passed_impact']/max(stats['passed_spatial'], 1)*100:.1f}%)")
    print(f"  Passed both filters: {stats['passed_both']} "
          f"({stats['passed_both']/stats['total_scenarios']*100:.1f}%)")
    
    stats["output_path"] = output_path
    return stats


def visualize_filtered_scenarios(library_path: str, max_scenarios: int = 10):
    """
    Visualize windstorm paths for filtered scenarios.

    Args:
        library_path: Path to the filtered library
        max_scenarios: Maximum number of scenarios to visualize
    """
    print(f"\n--- Visualizing Filtered Scenarios ---")

    # Load the filtered library
    library = load_library(library_path)
    scenarios = library.get("scenarios", {})

    if not scenarios:
        print("No scenarios to visualize.")
        return

    # Get scenario IDs (sorted for consistency)
    scenario_ids = sorted(list(scenarios.keys()))
    num_to_vis = min(len(scenario_ids), max_scenarios)

    print(f"Visualizing {num_to_vis} out of {len(scenario_ids)} filtered scenarios...")

    # Create temporary data with list structure for visualization
    temp_data = {
        "metadata": library.get("metadata", {}),
        "scenarios": [scenarios[sid] for sid in scenario_ids[:num_to_vis]]  # Convert to list
    }

    # Save temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(temp_data, f)
        temp_path = f.name

    try:
        # Visualize each scenario
        for i in range(num_to_vis):
            scenario_id = scenario_ids[i]
            scenario = scenarios[scenario_id]
            num_events = len(scenario.get("events", []))

            print(f"\nVisualizing {scenario_id} ({num_events} events)...")

            for event_idx in range(num_events):
                try:
                    visualize_windstorm_event(
                        file_path=temp_path,
                        scenario_number=i + 1,  # Now this works with list index
                        event_number=event_idx + 1,
                        custom_title=f"Filtered Scenario {scenario_id}, Event {event_idx + 1}"
                    )
                except Exception as e:
                    print(f"  Warning: Could not visualize event {event_idx + 1}: {e}")

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


def analyze_screened_library(library_path: str):
    """Analyze a screened library to understand impact patterns."""
    library = load_library(library_path)
    
    print(f"\nAnalyzing screened library: {Path(library_path).name}")
    print(f"Total scenarios: {len(library['scenarios'])}")
    
    # Analyze impact patterns
    total_events = 0
    impact_durations = []
    start_times = []
    
    for scenario in library["scenarios"].values():
        events = scenario.get("events", [])
        total_events += len(events)
        
        for event in events:
            impact_durations.append(event["duration"])
            start_times.append(event["bgn_hr"])
    
    if total_events > 0:
        print(f"\nEvent Statistics:")
        print(f"  Total windstorm events: {total_events}")
        print(f"  Average events per scenario: {total_events/len(library['scenarios']):.2f}")
        print(f"  Average event duration: {np.mean(impact_durations):.1f} hours")
        print(f"  Duration range: {min(impact_durations)}-{max(impact_durations)} hours")
        print(f"  Start time range: hour {min(start_times)}-{max(start_times)} of year")
        
        # Show screening info if available
        screening_info = library["metadata"].get("screening_info", {})
        if screening_info:
            print(f"\nScreening Information:")
            print(f"  Original scenarios: {screening_info.get('original_scenarios', 'N/A')}")
            print(f"  Filtered scenarios: {screening_info.get('filtered_scenarios', 'N/A')}")
            print(f"  Buffer: {screening_info.get('buffer_km', 'N/A')} km")
            print(f"  Min impacted branches: {screening_info.get('min_impacted_branches', 'N/A')}")
            print(f"  Min impact hours: {screening_info.get('min_impact_hours', 'N/A')}")


if __name__ == "__main__":
    # Example usage
    
    # Screen the library
    results = screen_scenarios(
        library_path="../Scenario_Database/Scenarios_Libraries/Original_Scenario_Libraries/windstorm_library_29BusGB-KearsleyGSP_GB_20scenarios_seed10000.json",
        output_dir="../Scenario_Database/Scenarios_Libraries/Filtered_Scenario_Libraries",
        buffer_km=15,
        min_impacted_branches=10,
        min_impact_hours=2,
        custom_dn_indices=None,
        custom_boundary=None
    )
    
    # Analyze the filtered library
    if results["passed_both"] > 0:
        analyze_screened_library(results["output_path"])
        
        # Visualize filtered scenarios (max 5 for demonstration)
        visualize_filtered_scenarios(results["output_path"], max_scenarios=5)
    else:
        print("\nNo scenarios passed both filters.")