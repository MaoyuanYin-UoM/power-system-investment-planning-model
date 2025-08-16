import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def load_results(results_file_path):
    """Load results from the JSON file generated after solving the model."""
    with open(results_file_path, 'r') as f:
        results = json.load(f)
    return results


def extract_power_values(results, scenario_indices,
                         network_name='29_bus_GB_transmission_network_with_Kearsley_GSP_group'):
    """
    Extract Pg (generation) and Pc (load shedding) values from results.
    Returns dictionaries indexed by (scenario, bus/gen, hour).
    """
    # Load network data to identify DN buses and generators
    from factories.network_factory import make_network
    net = make_network(network_name)

    # Get DN buses and generators
    dn_buses = [b for b in net.data.net.bus if net.data.net.bus_level[b] == 'D']
    dn_gens = [g for g in range(1, len(net.data.net.gen) + 1)
               if net.data.net.bus_level[net.data.net.gen[g - 1]] == 'D']

    # Extract Pg and Pc values
    pg_values = {}
    pc_values = {}

    # Parse the results structure
    # Note: You'll need to adjust this based on your exact results file structure
    for var_name, var_data in results.get('variables', {}).items():
        if var_name == 'Pg':
            for key, value in var_data.items():
                # Parse the key tuple (scenario, gen, hour)
                if isinstance(key, str):
                    # If key is string representation, parse it
                    key_parts = key.strip('()').split(',')
                    scenario = int(key_parts[0])
                    gen = int(key_parts[1])
                    hour = int(key_parts[2])
                else:
                    scenario, gen, hour = key

                # Only include DN generators
                if gen in dn_gens and scenario in scenario_indices:
                    pg_values[(scenario, gen, hour)] = value

        elif var_name == 'Pc':
            for key, value in var_data.items():
                if isinstance(key, str):
                    key_parts = key.strip('()').split(',')
                    scenario = int(key_parts[0])
                    bus = int(key_parts[1])
                    hour = int(key_parts[2])
                else:
                    scenario, bus, hour = key

                # Only include DN buses
                if bus in dn_buses and scenario in scenario_indices:
                    pc_values[(scenario, bus, hour)] = value

    return pg_values, pc_values, dn_buses, dn_gens


def calculate_resilience_curve(pg_values, pc_values, scenario_indices, dn_buses, dn_gens):
    """
    Calculate the resilience curve (expected energy supplied over time).
    """
    # Find the maximum number of hours across all scenarios
    max_hours = 0
    hours_per_scenario = {}

    for scenario in scenario_indices:
        scenario_hours = set()
        for key in pg_values.keys():
            if key[0] == scenario:
                scenario_hours.add(key[2])
        hours_per_scenario[scenario] = max(scenario_hours) if scenario_hours else 0
        max_hours = max(max_hours, hours_per_scenario[scenario])

    # Calculate energy supplied for each hour and scenario
    energy_supplied = np.zeros((len(scenario_indices), max_hours))

    for i, scenario in enumerate(scenario_indices):
        for hour in range(1, hours_per_scenario[scenario] + 1):
            # Total generation at DN level
            total_pg = sum(pg_values.get((scenario, g, hour), 0) for g in dn_gens)

            # Total load shedding at DN level
            total_pc = sum(pc_values.get((scenario, b, hour), 0) for b in dn_buses)

            # Energy supplied = generation - load shedding
            energy_supplied[i, hour - 1] = total_pg - total_pc

    # Calculate expected (average) energy supplied across scenarios
    # Assuming equal probability for each scenario
    expected_energy_supplied = np.mean(energy_supplied, axis=0)

    # Also calculate the energy supplied for each individual scenario
    return expected_energy_supplied, energy_supplied, max_hours


def plot_resilience_curve(expected_energy_supplied, energy_supplied, scenario_indices,
                          save_path=None, show_individual_scenarios=True):
    """
    Plot the resilience curve showing energy supplied over time.
    """
    hours = np.arange(1, len(expected_energy_supplied) + 1)

    plt.figure(figsize=(12, 8))

    # Plot individual scenarios if requested
    if show_individual_scenarios:
        for i, scenario in enumerate(scenario_indices):
            plt.plot(hours, energy_supplied[i, :],
                     alpha=0.3, linewidth=1,
                     label=f'Scenario {scenario}')

    # Plot expected (average) curve
    plt.plot(hours, expected_energy_supplied,
             color='black', linewidth=3,
             label='Expected Energy Supplied',
             marker='o', markevery=max(1, len(hours) // 20))

    # Add resilience metrics
    min_energy = np.min(expected_energy_supplied)
    max_energy = np.max(expected_energy_supplied)

    # Add horizontal lines for reference
    plt.axhline(y=max_energy, color='green', linestyle='--', alpha=0.5,
                label=f'Pre-event level: {max_energy:.1f} MW')
    plt.axhline(y=min_energy, color='red', linestyle='--', alpha=0.5,
                label=f'Minimum level: {min_energy:.1f} MW')

    # Highlight different phases (you can adjust these based on your specific case)
    # Event progression phase
    event_start = 0
    event_peak = np.argmin(expected_energy_supplied)
    recovery_start = event_peak
    recovery_end = len(hours) - 1

    plt.axvspan(event_start, event_peak, alpha=0.1, color='red',
                label='Event Progression')
    plt.axvspan(recovery_start, recovery_end, alpha=0.1, color='green',
                label='Recovery Phase')

    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Energy Supplied at Distribution Level (MW)', fontsize=12)
    plt.title('Resilience Curve During HILP Events', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)

    # Add annotation for resilience loss
    resilience_loss = (max_energy - min_energy) / max_energy * 100
    plt.text(0.02, 0.02, f'Max resilience loss: {resilience_loss:.1f}%',
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Resilience curve saved to: {save_path}")

    plt.show()

    return plt.gcf()


def calculate_resilience_metrics(expected_energy_supplied, time_hours):
    """
    Calculate various resilience metrics from the curve.
    """
    max_supply = np.max(expected_energy_supplied)
    min_supply = np.min(expected_energy_supplied)

    # Expected Energy Not Supplied (EENS)
    eens = np.sum(max_supply - expected_energy_supplied)

    # Time to minimum supply
    time_to_min = time_hours[np.argmin(expected_energy_supplied)]

    # Recovery time (time to return to 95% of pre-event level)
    recovery_threshold = 0.95 * max_supply
    recovery_indices = np.where(expected_energy_supplied >= recovery_threshold)[0]
    if len(recovery_indices) > 1:
        # Find first index after the minimum
        min_index = np.argmin(expected_energy_supplied)
        recovery_indices_after_min = recovery_indices[recovery_indices > min_index]
        if len(recovery_indices_after_min) > 0:
            recovery_time = time_hours[recovery_indices_after_min[0]] - time_hours[min_index]
        else:
            recovery_time = None
    else:
        recovery_time = None

    metrics = {
        'max_supply_mw': max_supply,
        'min_supply_mw': min_supply,
        'max_reduction_percent': (max_supply - min_supply) / max_supply * 100,
        'eens_mwh': eens,
        'time_to_min_hours': time_to_min,
        'recovery_time_hours': recovery_time
    }

    return metrics


# Main function to create resilience curve
def create_resilience_curve_from_results(results_file_path, scenario_indices=None,
                                         output_dir='Resilience_Analysis'):
    """
    Main function to create resilience curve from optimization results.

    Args:
        results_file_path: Path to the results JSON file
        scenario_indices: List of windstorm scenario indices (if None, uses all)
        output_dir: Directory to save the output figures
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from: {results_file_path}")
    results = load_results(results_file_path)

    # If scenario indices not specified, extract from results
    if scenario_indices is None:
        # Assuming scenarios are numbered 1, 2, 3, 4 for 4 windstorm scenarios
        scenario_indices = [1, 2, 3, 4]  # Adjust based on your scenario numbering

    # Extract power values
    print("Extracting power values...")
    pg_values, pc_values, dn_buses, dn_gens = extract_power_values(results, scenario_indices)

    # Calculate resilience curve
    print("Calculating resilience curve...")
    expected_energy, energy_by_scenario, max_hours = calculate_resilience_curve(
        pg_values, pc_values, scenario_indices, dn_buses, dn_gens
    )

    # Plot resilience curve
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f'resilience_curve_{timestamp}.png')

    print("Plotting resilience curve...")
    plot_resilience_curve(expected_energy, energy_by_scenario, scenario_indices,
                          save_path=save_path, show_individual_scenarios=True)

    # Calculate and print resilience metrics
    time_hours = np.arange(1, max_hours + 1)
    metrics = calculate_resilience_metrics(expected_energy, time_hours)

    print("\nResilience Metrics:")
    print(f"  Maximum supply: {metrics['max_supply_mw']:.2f} MW")
    print(f"  Minimum supply: {metrics['min_supply_mw']:.2f} MW")
    print(f"  Maximum reduction: {metrics['max_reduction_percent']:.1f}%")
    print(f"  Expected Energy Not Supplied (EENS): {metrics['eens_mwh']:.2f} MWh")
    print(f"  Time to minimum: {metrics['time_to_min_hours']} hours")
    if metrics['recovery_time_hours']:
        print(f"  Recovery time: {metrics['recovery_time_hours']} hours")

    # Save metrics to file
    metrics_path = os.path.join(output_dir, f'resilience_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")

    return expected_energy, metrics


# Example usage
if __name__ == "__main__":
    # Specify your results file path
    results_file = '../Optimization_Results/Investment_Model/results_network_29BusGB-Kearsley_4_ws_seed_[112, 152, 166, 198]_resilience_threshold_2.00e3_20250624_142239.xlsx'

    # Create resilience curve
    create_resilience_curve_from_results(
        results_file_path=results_file,
        scenario_indices=[1, 2, 3, 4],  # Adjust based on your scenario numbering
        output_dir='../Resilience_Analysis'
    )