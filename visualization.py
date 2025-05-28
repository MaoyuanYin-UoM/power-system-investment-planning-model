# This script contains function for visualization

from network import NetworkClass
from windstorm import WindClass
from network_factory import make_network
from windstorm_factory import make_windstorm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import lognorm
import json
import math

from windstorm_factory import make_windstorm


def visualize_ws_contour(windstorm_name: str = "default"):
    """Visualize the starting- and ending-points contour for windstorm path generation"""
    # load windstorm model
    if windstorm_name == 'default':
        ws = WindClass()
    else:
        ws = make_windstorm(windstorm_name)

    start_lon = ws.data.WS.contour.start_lon
    start_lat = ws.data.WS.contour.start_lat
    start_concty = ws.data.WS.contour.start_connectivity
    end_lon = ws.data.WS.contour.end_lon
    end_lat_coef = ws.data.WS.contour.end_lat_coef

    # Compute the upper and lower bounds for ending-point latitude
    end_lat = [end_lat_coef[0] * x + end_lat_coef[1] for x in end_lon]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot contours
    # 1) plot starting-point contour
    ax.scatter(start_lon, start_lat, color='blue', label='Starting Points', zorder=2)
    for connection in start_concty:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        ax.plot([start_lon[idx1], start_lon[idx2]], [start_lat[idx1], start_lat[idx2]], 'b-', alpha=0.7, zorder=1)

    # 2) plot ending-point contour
    ax.scatter(end_lon, end_lat, color='red', label='Ending Points', zorder=2)
    ax.plot(end_lon, end_lat, 'r-', alpha=0.7, zorder=1)

    # Labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Windstorm Starting and Ending Contours")
    ax.legend()
    ax.grid(True)

    plt.show()


def visualize_fragility_curve(WindConfig):
    wcon = WindConfig

    # Get parameters from config
    mu = wcon.data.frg.mu
    sigma = wcon.data.frg.sigma
    thrd_1 = wcon.data.frg.thrd_1
    thrd_2 = wcon.data.frg.thrd_2
    shift_f = wcon.data.frg.shift_f

    # Generate hazard intensities
    hzd_int_range = np.linspace(0, 120, 500)  # Hazard intensity from 0 to 120
    pof_values = []

    # Calculate PoF for each hazard intensity
    for hzd_int in hzd_int_range:
        f_hzd_int = hzd_int - shift_f
        if f_hzd_int < thrd_1:
            pof = 0
        elif f_hzd_int > thrd_2:
            pof = 1
        else:
            shape = sigma
            scale = np.exp(mu)
            pof = lognorm.cdf(f_hzd_int, s=shape, scale=scale)
        pof_values.append(pof)

    # Plot the fragility curve
    plt.figure(figsize=(8, 6))
    plt.plot(hzd_int_range, pof_values, label='Fragility Curve')
    plt.axvline(thrd_1, color='green', linestyle='--', label=f'Threshold 1 ({thrd_1})')
    plt.axvline(thrd_2, color='red', linestyle='--', label=f'Threshold 2 ({thrd_2})')
    plt.xlabel('Hazard Intensity (e.g., Wind Speed)')
    plt.ylabel('Probability of Failure (PoF)')
    plt.title('Fragility Curve')
    plt.legend()
    plt.grid()
    plt.show()


def visualize_bch_and_ws_contour(network_name: str = "default", windstorm_name: str = "default"):
    """
    Visualize the branches along with the starting- and ending-points contour for windstorm path generation.

    Parameters:
    - WindConfig: Configuration object containing windstorm data.
    - NetworkConfig: Configuration object containing network data.
    """
    from network import NetworkClass
    from windstorm import WindClass

    # load network model
    if network_name == 'default':
        net = NetworkClass()
    else:
        net = make_network(network_name)

    # load windstorm model
    if windstorm_name == 'default':
        ws = WindClass()
    else:
        ws = make_windstorm(windstorm_name)

    # Windstorm data
    start_lon = ws.data.WS.contour.start_lon
    start_lat = ws.data.WS.contour.start_lat
    start_concty = ws.data.WS.contour.start_connectivity
    end_lon = ws.data.WS.contour.end_lon
    end_lat_coef = ws.data.WS.contour.end_lat_coef
    end_lat = [end_lat_coef[0] * x + end_lat_coef[1] for x in end_lon]

    # Branch data
    net.set_gis_data()
    bch_gis_bgn = net._get_bch_gis_bgn()
    bch_gis_end = net._get_bch_gis_end()

    # Validate branch data
    if not bch_gis_bgn or not bch_gis_end:
        print("Error: Branch GIS data is missing or invalid.")
        return

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot windstorm contours
    ax.scatter(start_lon, start_lat, color='blue', label='Starting Points', zorder=2)
    for connection in start_concty:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        ax.plot([start_lon[idx1], start_lon[idx2]], [start_lat[idx1], start_lat[idx2]], 'b-', alpha=0.7, zorder=1)

    ax.scatter(end_lon, end_lat, color='red', label='Ending Points', zorder=2)
    ax.plot(end_lon, end_lat, 'r-', alpha=0.7, zorder=1)

    # Plot branches
    for bgn, end in zip(bch_gis_bgn, bch_gis_end):
        ax.plot([bgn[0], end[0]], [bgn[1], end[1]], 'g-', alpha=0.8, zorder=1, label='Branch' if bgn == bch_gis_bgn[0] else "")

    # Set axis limits
    ax.relim()
    ax.autoscale()

    # Labels, title, and legend
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Branches and Windstorm Contours")
    ax.legend()
    ax.grid(True)

    plt.show()



def visualize_windstorm_event(file_path, scenario_number, event_number):
    """
    Visualizes the path of a windstorm event from the stored ws_scenarios .json files.
    """
    # Load data from the .json file
    with open(file_path, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    if "scenarios" in data:
        scenarios = data["scenarios"]
    elif "ws_scenarios" in data:
        scenarios = data["ws_scenarios"]

    # Instantiate network and windstorm based on the metadata
    net = make_network(meta["network_preset"])
    ws = make_windstorm(meta["windstorm_preset"])

    # Select scenario
    scenario_idx = scenario_number - 1  # Convert 1-based to 0-based index
    if scenario_idx >= len(scenarios):
        print(f"Scenario {scenario_number} not found!")
        return

    scenario = scenarios[scenario_idx]

    # Select event
    event_idx = event_number - 1
    events = scenario.get("events", [])
    if event_idx >= len(events):
        print(f"Event {event_number} not found in Scenario {scenario_number}!")
        return

    event = events[event_idx]

    # Extract windstorm data
    epicentres = np.array(event["epicentre"])  # Convert to NumPy array
    radius_km = event["radius"]  # Windstorm radius at each timestep

    # Convert radius from km to degrees for plotting
    radius_deg = []
    for i, (lon, lat) in enumerate(epicentres):
        lat_factor = 111  # Assumption: 1 degree latitude ≈ 111 km
        lon_factor = 111 * math.cos(math.radians(lat))  # Longitude factor varies with latitude

        # Convert km to degrees using latitude scaling
        r_deg = radius_km[i] / lat_factor  # Using latitude for scaling
        radius_deg.append(r_deg)

    # Extract branch data from network
    net.set_gis_data()  # Ensure GIS data is set
    bch_gis_bgn = net.data.net.bch_gis_bgn
    bch_gis_end = net.data.net.bch_gis_end

    # Plot the network branches
    fig, ax = plt.subplots(figsize=(10, 8))
    for bgn, end in zip(bch_gis_bgn, bch_gis_end):
        ax.plot([bgn[0], end[0]], [bgn[1], end[1]], 'g-', alpha=0.7, label="Branch" if bgn == bch_gis_bgn[0] else "")

    # Plot the windstorm path
    ax.plot(epicentres[:, 0], epicentres[:, 1], 'bo-', label="Windstorm Path", alpha=0.8)

    # Plot epicentres and circles for each timestep
    for i, (lon, lat) in enumerate(epicentres):
        ax.scatter(lon, lat, color="blue", s=40, zorder=3)  # Epicentre
        circle = Circle((lon, lat), radius_deg[i], color='blue', alpha=0.2, fill=True)
        ax.add_patch(circle)  # Windstorm radius

    # Set axis limits (i.e., there's always a 2-unit margin beyond the min/max coordinates of the network model)
    bus_lons = net._get_bus_lon()
    bus_lats = net._get_bus_lat()
    xmin, xmax = min(bus_lons), max(bus_lons)
    ymin, ymax = min(bus_lats), max(bus_lats)
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)

    # Labels, title, and legend
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Windstorm Path - Scenario {scenario_number}, Event {event_number}")
    ax.legend()
    ax.grid(True)

    plt.show()


def visualize_all_windstorm_events(
    file_path="Scenario_Results/all_full_scenarios_year.json"
):
    """
    Loop through all scenarios/events in the given JSON and
    visualize each windstorm event in its own figure.
    """
    # Load data from the .json file
    with open(file_path, "r") as f:
        data = json.load(f)

    if "scenarios" in data:
        scenarios = data["scenarios"]
    elif "ws_scenarios" in data:
        scenarios = data["ws_scenarios"]

    # 2) Loop and plot
    for scen_idx, scenario in enumerate(scenarios, start=1):
        num_events = len(scenario.get("events", []))
        if num_events == 0:
            print(f"Scenario {scen_idx} has no events, skipping.")
            continue

        for ev_idx in range(1, num_events+1):
            print(f"Visualizing Scenario {scen_idx}, Event {ev_idx}...")
            visualize_windstorm_event(
                file_path=file_path,
                scenario_number=scen_idx,
                event_number=ev_idx
            )