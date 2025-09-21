# ws_scenario_clustering.py
"""
Compute EENS for scenarios and cluster them based on EENS impact.
"""

import json
import numpy as np
from sklearn.cluster import KMeans
import os
from datetime import datetime
from factories.network_factory import make_network


def compute_eens_for_scenarios(
        scenario_library_path: str,
        network_name: str = "29_bus_GB_transmission_network_with_Kearsley_GSP_group",
        solver: str = "gurobi",
        output_dir: str = None):
    """
    Compute EENS for all scenarios and save enhanced library with EENS values.

    Args:
        scenario_library_path: Path to scenario library (filtered or original)
        network_name: Network preset name
        solver: Solver to use for OPF
        output_dir: Optional output directory (default: EENS_Enhanced_Scenario_Libraries)

    Returns:
        Path to the enhanced library with EENS values
    """

    print(f"\n{'=' * 60}")
    print("COMPUTING EENS FOR SCENARIOS")
    print(f"{'=' * 60}")

    # Load scenarios
    with open(scenario_library_path, 'r') as f:
        ws_library = json.load(f)

    scenarios = ws_library.get("scenarios", {})
    metadata = ws_library.get("metadata", {})
    scenario_ids = list(scenarios.keys())

    print(f"Loaded {len(scenario_ids)} scenarios from library")
    print(f"Network: {network_name}")
    print(f"Solver: {solver}")

    # Initialize network
    net = make_network(network_name)

    # Compute EENS for each scenario
    print("\nComputing EENS for each scenario...")
    eens_values = {}
    failed_scenarios = []

    for idx, scenario_id in enumerate(scenario_ids, 1):
        print(f"  Processing {idx}/{len(scenario_ids)}: {scenario_id}", end="")

        try:
            # Build and solve OPF model
            model = net.build_combined_opf_model_under_ws_scenarios(
                single_ws_scenario=scenarios[scenario_id]
            )

            eens = net.solve_combined_opf_model_under_ws_scenarios(
                model=model,
                solver_name=solver,
                mip_gap=5e-3,
                time_limit=60
            )

            eens_values[scenario_id] = float(eens)
            print(f" -> EENS: {eens:.2f} MWh")

        except Exception as e:
            print(f" -> FAILED: {str(e)[:50]}...")
            failed_scenarios.append(scenario_id)
            continue

    # Print summary
    print(f"\n{'=' * 40}")
    print("EENS COMPUTATION SUMMARY:")
    print(f"  Successfully computed: {len(eens_values)}/{len(scenario_ids)}")
    print(f"  Failed scenarios: {len(failed_scenarios)}")

    if eens_values:
        eens_list = list(eens_values.values())
        print(f"  EENS range: {min(eens_list):.2f} - {max(eens_list):.2f} MWh")
        print(f"  Average EENS: {np.mean(eens_list):.2f} MWh")
        print(f"  Std EENS: {np.std(eens_list):.2f} MWh")

    # Create enhanced library with EENS data
    enhanced_library = {
        "metadata": {
            **metadata,  # Keep original metadata
            "eens_computation": {
                "source_library": os.path.basename(scenario_library_path),
                "computation_date": datetime.now().isoformat(),
                "network": network_name,
                "solver": solver,
                "total_scenarios": len(scenario_ids),
                "computed_scenarios": len(eens_values),
                "failed_scenarios": len(failed_scenarios)
            }
        },
        "scenarios": scenarios,  # Keep original scenarios
        "eens_values": eens_values,
        "failed_scenarios": failed_scenarios
    }

    # Generate output path
    if output_dir is None:
        # Default: replace directory with EENS_Enhanced_Scenario_Libraries
        input_dir = os.path.dirname(scenario_library_path)
        base_dir = os.path.dirname(input_dir)  # Go up one level
        output_dir = os.path.join(base_dir, "EENS_Enhanced_Scenario_Libraries")

    # Create output filename (append _eens_added)
    input_filename = os.path.basename(scenario_library_path)
    name_without_ext = os.path.splitext(input_filename)[0]
    output_filename = f"{name_without_ext}_eens.json"
    output_path = os.path.join(output_dir, output_filename)

    # Save enhanced library
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(enhanced_library, f, indent=2)

    print(f"\nEnhanced library saved to: {output_path}")
    print(f"{'=' * 40}")

    return output_path


def cluster_scenarios_by_eens(
        eens_enhanced_library_path: str,
        n_clusters: int = 6,
        output_dir: str = None):
    """
    Cluster scenarios using pre-computed EENS values from enhanced library.

    Args:
        eens_enhanced_library_path: Path to EENS-enhanced scenario library
        n_clusters: Number of representative scenarios to select
        output_dir: Optional output directory (default: Clustered_Scenario_Libraries)

    Returns:
        Tuple of (selected_scenarios, scenario_probabilities)
    """

    print(f"\n{'=' * 60}")
    print("CLUSTERING SCENARIOS BY EENS")
    print(f"{'=' * 60}")
    print(f"Target clusters: {n_clusters}")

    # Load enhanced library
    with open(eens_enhanced_library_path, 'r') as f:
        enhanced_lib = json.load(f)

    scenarios = enhanced_lib.get("scenarios", {})
    metadata = enhanced_lib.get("metadata", {})
    eens_values_dict = enhanced_lib.get("eens_values", {})
    failed_scenarios = enhanced_lib.get("failed_scenarios", [])

    if not eens_values_dict:
        print("Error: No EENS values found in library!")
        return {}, {}

    print(f"Loaded {len(scenarios)} scenarios")
    print(f"Available EENS values: {len(eens_values_dict)}")
    print(f"Failed scenarios: {len(failed_scenarios)}")

    # Prepare data for clustering
    valid_scenario_ids = list(eens_values_dict.keys())
    eens_array = np.array([[eens_values_dict[sid]] for sid in valid_scenario_ids])

    # Check if we have enough scenarios
    if len(valid_scenario_ids) < n_clusters:
        print(f"Warning: Only {len(valid_scenario_ids)} valid scenarios, reducing clusters")
        n_clusters = len(valid_scenario_ids)

    # Perform K-means clustering
    print(f"\nPerforming K-means clustering (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(eens_array)

    # Select representative scenarios
    print("\nSelecting representative scenarios...")
    selected_scenarios = {}
    scenario_probabilities = {}
    cluster_info = []

    for cluster_id in range(n_clusters):
        # Find scenarios in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        # Find scenario closest to cluster center
        center_eens = kmeans.cluster_centers_[cluster_id][0]
        cluster_eens_values = eens_array[cluster_indices].flatten()
        distances = np.abs(cluster_eens_values - center_eens)
        closest_idx = cluster_indices[np.argmin(distances)]

        # Get selected scenario
        selected_id = valid_scenario_ids[closest_idx]
        selected_eens = eens_array[closest_idx][0]

        # Calculate probability
        probability = len(cluster_indices) / len(valid_scenario_ids)

        # Store results
        selected_scenarios[selected_id] = scenarios[selected_id]
        scenario_probabilities[selected_id] = probability

        cluster_info.append({
            "cluster_id": cluster_id,
            "representative": selected_id,
            "eens_mwh": float(selected_eens),
            "probability": probability,
            "cluster_size": len(cluster_indices),
            "cluster_center_eens": float(center_eens),
            "cluster_eens_range": [float(cluster_eens_values.min()),
                                   float(cluster_eens_values.max())]
        })

        print(f"  Cluster {cluster_id}: {selected_id} "
              f"(EENS={selected_eens:.2f} MWh, prob={probability:.3f})")

    # Check clustering quality
    print(f"\n{'=' * 40}")
    print("CLUSTERING QUALITY CHECK:")
    original_avg = np.mean(eens_array)
    weighted_avg = sum(cluster_info[i]["eens_mwh"] * cluster_info[i]["probability"]
                       for i in range(n_clusters))
    percent_diff = ((weighted_avg - original_avg) / original_avg) * 100

    print(f"  Original average EENS: {original_avg:.2f} MWh")
    print(f"  Clustered weighted avg: {weighted_avg:.2f} MWh")
    print(f"  Relative difference: {percent_diff:+.2f}%")

    # Create output data
    output_data = {
        "metadata": {
            **metadata,
            "clustering_info": {
                "source_library": os.path.basename(eens_enhanced_library_path),
                "clustering_method": "kmeans_eens",
                "n_clusters": n_clusters,
                "total_scenarios": len(scenarios),
                "valid_scenarios_used": len(valid_scenario_ids),
                "clustering_date": datetime.now().isoformat(),
                "clustering_quality": {
                    "original_avg_eens": float(original_avg),
                    "clustered_weighted_avg": float(weighted_avg),
                    "relative_difference_percent": float(percent_diff)
                }
            }
        },
        "scenarios": selected_scenarios,
        "scenario_probabilities": scenario_probabilities,
        "cluster_details": cluster_info
    }

    # Generate output path
    if output_dir is None:
        # Default: Clustered_Scenario_Libraries
        input_dir = os.path.dirname(eens_enhanced_library_path)
        base_dir = os.path.dirname(input_dir)
        output_dir = os.path.join(base_dir, "Clustered_Scenario_Libraries")

    # Create output filename (remove _eens_added, add _k{n})
    input_filename = os.path.basename(eens_enhanced_library_path)
    name_without_ext = os.path.splitext(input_filename)[0]
    name_without_eens = name_without_ext.replace("_eens_added", "")
    output_filename = f"{name_without_eens}_k{n_clusters}.json"
    output_path = os.path.join(output_dir, output_filename)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nClustered scenarios saved to: {output_path}")
    print(f"{'=' * 40}")

    return selected_scenarios, scenario_probabilities


def main():
    """
    Example workflow: compute EENS then cluster.
    """

    # Step 1: Compute EENS for filtered scenarios
    filtered_path = "../Scenario_Database/Scenarios_Libraries/Filtered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_29GB_1000scn_s50000_filt_b1_h1_buf15.json"
    enhanced_path = "../Scenario_Database/Scenarios_Libraries/EENS_Enhanced_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_29GB_1000scn_s50000_filt_b1_h1_buf15_eens.json"

    print("STEP 1: Computing EENS for scenarios...")
    enhanced_path = compute_eens_for_scenarios(
        scenario_library_path=filtered_path,
        network_name="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
        solver="gurobi"
    )

    # Step 2: Cluster using pre-computed EENS
    print("\nSTEP 2: Clustering scenarios...")
    selected_scenarios, probabilities = cluster_scenarios_by_eens(
        eens_enhanced_library_path=enhanced_path,
        n_clusters=10
    )

    print(f"\n{'=' * 60}")
    print("COMPLETE WORKFLOW FINISHED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()