# ws_scenario_clustering.py
"""
Cluster windstorm scenarios based on EENS impact to select representative scenarios.
"""

import json
import numpy as np
from sklearn.cluster import KMeans
import os
from datetime import datetime
from factories.network_factory import make_network


def cluster_scenarios_by_eens(
        screened_library_path: str,
        n_clusters: int = 6,
        network_name: str = "29_bus_GB_transmission_network_with_Kearsley_GSP_group",
        solver: str = "gurobi"):
    """
    Cluster scenarios based on their EENS impact and select representatives.

    Args:
        screened_library_path: Path to filtered windstorm scenario library
        n_clusters: Number of representative scenarios to select
        network_name: Network preset name
        solver: Solver to use for OPF

    Returns:
        Dictionary of selected scenarios with probabilities
    """

    print(f"\nClustering windstorm scenarios by EENS...")
    print(f"Target clusters: {n_clusters}")

    # Load scenarios
    with open(screened_library_path, 'r') as f:
        ws_library = json.load(f)

    scenarios = ws_library.get("scenarios", {})
    metadata = ws_library.get("metadata", {})
    scenario_ids = list(scenarios.keys())

    print(f"Loaded {len(scenario_ids)} scenarios")

    # Initialize network
    net = make_network(network_name)

    # Compute EENS for each scenario
    print("\nComputing EENS for each scenario...")
    eens_values = []
    valid_scenario_ids = []
    skipped_scenarios = []

    for idx, scenario_id in enumerate(scenario_ids, 1):
        print(f"  Processing {idx}/{len(scenario_ids)}: {scenario_id}")

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

            eens_values.append([eens])  # 2D array for sklearn
            valid_scenario_ids.append(scenario_id)
            print(f"    EENS: {eens:.2f} MWh")

        except Exception as e:
            print(f"    SKIPPED - Error: {str(e)[:100]}...")  # Print first 100 chars of error
            skipped_scenarios.append(scenario_id)
            continue

    # Check if we have enough valid scenarios
    if len(valid_scenario_ids) < n_clusters:
        print(f"\nWarning: Only {len(valid_scenario_ids)} valid scenarios, reducing clusters to this number")
        n_clusters = len(valid_scenario_ids)

    if len(valid_scenario_ids) == 0:
        print("\nError: No valid scenarios found!")
        return {}, {}

    # Convert to numpy array
    eens_array = np.array(eens_values)

    print(f"\nScenario processing summary:")
    print(f"  Valid scenarios: {len(valid_scenario_ids)}")
    print(f"  Skipped scenarios: {len(skipped_scenarios)}")
    if skipped_scenarios:
        print(f"  Skipped IDs: {skipped_scenarios[:10]}...")  # Show first 10

    print(f"\nEENS statistics for valid scenarios:")
    print(f"  Min: {eens_array.min():.2f} MWh")
    print(f"  Max: {eens_array.max():.2f} MWh")
    print(f"  Mean: {eens_array.mean():.2f} MWh")

    # Perform K-means clustering
    print(f"\nPerforming K-means clustering (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(eens_array)

    # Select representative scenarios (closest to each cluster center)
    print("\nSelecting representative scenarios...")
    selected_scenarios = {}
    scenario_probabilities = {}

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

        # Calculate probability (proportion of scenarios in cluster)
        probability = len(cluster_indices) / len(valid_scenario_ids)

        # Store selected scenario (keep original format)
        selected_scenarios[selected_id] = scenarios[selected_id]
        scenario_probabilities[selected_id] = probability

        print(f"  Cluster {cluster_id}: {selected_id} "
              f"(EENS={selected_eens:.2f} MWh, prob={probability:.3f})")

    # Create output data in same format as input library
    output_data = {
        "metadata": {
            **metadata,  # Keep all original metadata
            "clustering_info": {
                "clustering_method": "kmeans_eens",
                "n_clusters": n_clusters,
                "original_scenarios_before_clustering": len(scenarios),
                "valid_scenarios_used": len(valid_scenario_ids),
                "skipped_scenarios": len(skipped_scenarios),
                "clustering_date": datetime.now().isoformat()
            }
        },
        "scenarios": selected_scenarios,  # Keep scenarios in original format
        "scenario_probabilities": scenario_probabilities  # Store probabilities separately
    }

    # Generate output path based on input filename
    output_path = generate_output_path(screened_library_path, n_clusters)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSelected scenarios saved to: {output_path}")

    return selected_scenarios, scenario_probabilities


def generate_output_path(input_path: str, n_clusters: int) -> str:
    """
    Generate output path with k value appended to filename.

    Example:
        Input:  ../Filtered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_GB_100scn_s10000_filt_b10_h2_buf15.json
        Output: ../Clustered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_GB_100scn_s10000_filt_b10_h2_buf15_k6.json
    """
    # Extract filename without extension
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(input_filename)[0]

    # Add k value to filename
    new_filename = f"{name_without_ext}_k{n_clusters}.json"

    # Replace directory path
    output_dir = input_dir.replace("Filtered_Scenario_Libraries", "Clustered_Scenario_Libraries")

    # If the replacement didn't work, use a default path
    if output_dir == input_dir:
        output_dir = "../Scenario_Database/Scenarios_Libraries/Clustered_Scenario_Libraries"

    output_path = os.path.join(output_dir, new_filename)

    return output_path


def main():
    """
    Example usage of the clustering function.
    """
    # Input path
    input_path = "../Scenario_Database/Scenarios_Libraries/Filtered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_GB_1000scn_s10000_filt_b1_h1_buf15.json"

    # Run clustering
    selected_scenarios, probabilities = cluster_scenarios_by_eens(
        screened_library_path=input_path,
        n_clusters=4
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("CLUSTERING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Selected {len(selected_scenarios)} representative scenarios")

    total_prob = sum(probabilities.values())
    print(f"Total probability: {total_prob:.4f}")

    for sid, prob in probabilities.items():
        print(f"  {sid}: Probability={prob:.3f}")


if __name__ == "__main__":
    main()