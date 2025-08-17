# ws_scenario_clustering_comparison.py
"""
Compare EENS between original filtered scenarios and clustered representatives.
"""

import json
import numpy as np


def compare_clustering_accuracy(
        eens_enhanced_library_path: str = None,
        clustered_library_path: str = None,
        filtered_library_path: str = None):
    """
    Compare average EENS between original and clustered scenarios.

    Args:
        eens_enhanced_library_path: Path to EENS-enhanced library (preferred)
        clustered_library_path: Path to clustered scenario library
        filtered_library_path: Optional, for backward compatibility
    """

    print("\n" + "=" * 60)
    print("CLUSTERING ACCURACY COMPARISON")
    print("=" * 60)

    # Load clustered library
    with open(clustered_library_path, 'r') as f:
        clustered_lib = json.load(f)

    # Get original EENS values
    original_eens_values = None

    # Method 1: From EENS-enhanced library (preferred)
    if eens_enhanced_library_path:
        print(f"Loading EENS from enhanced library...")
        with open(eens_enhanced_library_path, 'r') as f:
            enhanced_lib = json.load(f)

        eens_dict = enhanced_lib.get("eens_values", {})
        if eens_dict:
            original_eens_values = list(eens_dict.values())
            print(f"  Found {len(original_eens_values)} EENS values")

    # Method 2: Check if clustering quality already computed
    clustering_info = clustered_lib["metadata"].get("clustering_info", {})
    clustering_quality = clustering_info.get("clustering_quality", {})

    if clustering_quality and "original_avg_eens" in clustering_quality:
        # Clustering already includes the comparison!
        print("\nUsing pre-computed clustering quality from clustered library:")
        original_avg = clustering_quality["original_avg_eens"]
        clustered_avg = clustering_quality["clustered_weighted_avg"]
        percent_diff = clustering_quality["relative_difference_percent"]

        print(f"  Original average EENS:  {original_avg:.2f} MWh")
        print(f"  Clustered weighted avg: {clustered_avg:.2f} MWh")
        print(f"  Relative difference:    {percent_diff:+.2f}%")

        # Quality assessment
        if abs(percent_diff) < 5:
            print(f"  ✓ Excellent clustering (< 5% difference)")
        elif abs(percent_diff) < 10:
            print(f"  ✓ Good clustering (< 10% difference)")
        elif abs(percent_diff) < 20:
            print(f"  ⚠ Acceptable clustering (< 20% difference)")
        else:
            print(f"  ✗ Poor clustering (> 20% difference)")

        print("=" * 60)
        return

    # Method 3: Compute from scratch if we have original EENS
    if original_eens_values:
        original_avg_eens = np.mean(original_eens_values)

        print(f"\nOriginal scenarios:")
        print(f"  Number of scenarios: {len(original_eens_values)}")
        print(f"  Average EENS: {original_avg_eens:.2f} MWh")
        print(f"  Min EENS: {min(original_eens_values):.2f} MWh")
        print(f"  Max EENS: {max(original_eens_values):.2f} MWh")
        print(f"  Std EENS: {np.std(original_eens_values):.2f} MWh")
    else:
        print("\nWarning: Original EENS values not found")
        print("Please provide path to EENS-enhanced library")
        original_avg_eens = None

    # Get clustered scenario info
    cluster_details = clustered_lib.get("cluster_details", [])
    probabilities = clustered_lib.get("scenario_probabilities", {})

    if cluster_details:
        # Use detailed cluster info
        print(f"\nClustered representative scenarios:")
        print(f"  Number of representatives: {len(cluster_details)}")

        weighted_sum = 0
        total_prob = 0

        for cluster in cluster_details:
            scenario_id = cluster["representative"]
            eens = cluster["eens_mwh"]
            prob = cluster["probability"]
            cluster_size = cluster["cluster_size"]

            weighted_sum += eens * prob
            total_prob += prob

            print(f"    {scenario_id}: EENS={eens:.2f} MWh, "
                  f"Prob={prob:.3f}, Cluster size={cluster_size}")

        clustered_weighted_avg = weighted_sum / total_prob if total_prob > 0 else 0

    else:
        print("\nWarning: Cluster details not found in clustered library")
        return

    # Final comparison
    if original_avg_eens is not None and clustered_weighted_avg is not None:
        difference = clustered_weighted_avg - original_avg_eens
        percent_diff = (difference / original_avg_eens) * 100 if original_avg_eens != 0 else 0

        print(f"\n" + "=" * 60)
        print("COMPARISON RESULTS:")
        print(f"  Original average EENS:  {original_avg_eens:.2f} MWh")
        print(f"  Clustered weighted avg: {clustered_weighted_avg:.2f} MWh")
        print(f"  Absolute difference:    {abs(difference):.2f} MWh")
        print(f"  Relative difference:    {percent_diff:+.2f}%")

        if abs(percent_diff) < 5:
            print(f"  ✓ Excellent clustering (< 5% difference)")
        elif abs(percent_diff) < 10:
            print(f"  ✓ Good clustering (< 10% difference)")
        elif abs(percent_diff) < 20:
            print(f"  ⚠ Acceptable clustering (< 20% difference)")
        else:
            print(f"  ✗ Poor clustering (> 20% difference)")

    print("=" * 60)


def main():
    """
    Example usage of comparison with new structure.
    """
    # New workflow: use EENS-enhanced library
    eens_path = "../Scenario_Database/Scenarios_Libraries/EENS_Enhanced_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_GB_1000scn_s10000_filt_b1_h1_buf15_eens.json"
    clustered_path = "../Scenario_Database/Scenarios_Libraries/Clustered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_GB_1000scn_s10000_filt_b1_h1_buf15_eens_k3.json"

    # Method 1: Provide EENS-enhanced library
    compare_clustering_accuracy(
        eens_enhanced_library_path=eens_path,
        clustered_library_path=clustered_path
    )

    # Method 2: Just use clustered library (if it has quality info)
    # compare_clustering_accuracy(
    #     clustered_library_path=clustered_path
    # )


if __name__ == "__main__":
    main()