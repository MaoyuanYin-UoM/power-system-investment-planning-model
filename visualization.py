# This script contains function for visualization

from network import NetworkClass
from windstorm import WindClass
from network_factory import make_network
from windstorm_factory import make_windstorm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import lognorm
import json
import math

from windstorm_factory import make_windstorm



def visualize_network_bch(network_name: str = "default",
                          ax=None,
                          color: str = "grey",
                          alpha: float = 0.8,
                          linewidth: float = 1.2,
                          label_buses: bool = True,
                          label_fontsize: int = 8,
                          label_offset_lon: float = 0.02,
                          label_color: str = "black"):
    """
    Quick map of all branches of `network_name`, with optional
    bus‐number labels.

    Parameters
    ----------
    network_name : str
        Name of the network preset registered in `network_factory`.
    ax : matplotlib.axes.Axes | None
        If given, draw on this axis; otherwise create a fresh
        figure/axis and show it before returning.
    color, alpha, linewidth : matplotlib styling knobs for branches.
    label_buses : bool, optional
        If True, write the bus ID next to each plotted node.
    label_fontsize : int
        Font size for the labels.
    label_offset_lon : float
        Horizontal offset (in degrees) applied to the label so it
        does not overlap the node marker.
    label_color : str
        Text colour of the bus labels.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the branches (and labels) were drawn on.
    """
    # ----- load network & GIS data ----------------------------------------
    net = NetworkClass() if network_name == "default" else make_network(network_name)
    net.set_gis_data()

    # branch end-points
    bgn = net._get_bch_gis_bgn()        # list[(lon, lat)]
    end = net._get_bch_gis_end()

    # bus points -- needed for labels
    bus_lon = net._get_bus_lon()        # list[float]
    bus_lat = net._get_bus_lat()        # list[float]
    bus_ids = net.data.net.bus          # list[int]

    # ----- prepare axis ----------------------------------------------------
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        created_fig = True

    # ----- draw every branch ----------------------------------------------
    for p, q in zip(bgn, end):
        ax.plot([p[0], q[0]], [p[1], q[1]],
                color=color, alpha=alpha, lw=linewidth)

    # ----- draw bus markers & labels --------------------------------------
    if label_buses:
        for lon, lat, bid in zip(bus_lon, bus_lat, bus_ids):
            # a tiny marker (optional -- comment out if cluttered)
            ax.scatter(lon, lat, s=10, c=label_color, zorder=3)
            # text label, slightly offset in longitude
            ax.text(lon + label_offset_lon, lat,
                    str(bid),
                    fontsize=label_fontsize,
                    ha="left", va="center",
                    color=label_color,
                    zorder=4)

    # ----- cosmetics -------------------------------------------------------
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Branches – {network_name}")
    ax.grid(True)

    # ----- show the figure if we created it -------------------------------
    if created_fig:
        plt.show()

    return ax


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
    plt.xlabel('Wind Speed')
    plt.ylabel('Failure Probability')
    plt.title('Fragility Curve')

    # Move legend to top right corner
    plt.legend(loc='upper left', frameon=True, shadow=True)

    plt.grid()
    plt.show()


def visualize_windstorm_event(file_path, scenario_number, event_number, custom_title=None):
    """
    Visualizes the path of a windstorm event from the stored ws_scenarios .json files.

    Parameters:
    - file_path: Path to the JSON file containing windstorm scenarios
    - scenario_number: Scenario number (1-based)
    - event_number: Event number (1-based)
    - custom_title: Custom title for the plot. If None, uses default format
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

    # Check if branch_level exists
    has_branch_levels = hasattr(net.data.net, 'branch_level')

    # Plot the network branches
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create flags for legend (to avoid duplicate entries)
    tn_plotted = False
    dn_plotted = False

    for idx, (bgn, end) in enumerate(zip(bch_gis_bgn, bch_gis_end)):
        # Determine branch level if available
        if has_branch_levels:
            branch_level = net.data.net.branch_level.get(idx + 1, 'T')  # Default to 'T' if not found
        else:
            branch_level = 'T'  # Default to transmission if no level info

        # Set color based on branch level
        if branch_level == 'T' or branch_level == 'T-D':
            color = 'darkgreen'
            label = 'Transmission Branch' if not tn_plotted else ""
            tn_plotted = True
        else:  # 'D'
            color = 'orange'
            label = 'Distribution Branch' if not dn_plotted else ""
            dn_plotted = True

        ax.plot([bgn[0], end[0]], [bgn[1], end[1]], color=color,
                alpha=0.7, label=label)

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

    # Use custom title if provided, otherwise use default
    if custom_title:
        ax.set_title(custom_title)
    else:
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

        for ev_idx in range(1, num_events + 1):
            print(f"Visualizing Scenario {scen_idx}, Event {ev_idx}...")
            visualize_windstorm_event(
                file_path=file_path,
                scenario_number=scen_idx,
                event_number=ev_idx
            )


def visualize_bch_and_ws_contour(network_name: str = "default",
                                 windstorm_name: str = "default",
                                 label_buses: bool = True,
                                 label_fontsize: int = 8,
                                 label_offset_lon: float = 0.02,
                                 label_color: str = "black",
                                 zoomed_distribution: bool = False,
                                 zoom_border: float = 0.1,
                                 tn_linewidth: float = 1.5,
                                 dn_linewidth: float = 1.5,
                                 title: str = None):  # New parameter
    """
    Visualize the branches along with the starting- and ending-points contour for windstorm path generation.
    Now includes bus ID labels similar to visualize_network_bch.

    Parameters:
    - network_name: Name of the network preset registered in network_factory
    - windstorm_name: Name of the windstorm preset registered in windstorm_factory
    - label_buses: If True, write the bus ID next to each plotted node
    - label_fontsize: Font size for the labels
    - label_offset_lon: Horizontal offset (in degrees) applied to the label
    - label_color: Text colour of the bus labels
    - zoomed_distribution: If True, zoom in to show distribution network details
    - zoom_border: Border size for zoomed view (in degrees)
    - tn_linewidth: Line thickness for transmission branches
    - dn_linewidth: Line thickness for distribution branches
    - title: Custom title for the plot. If None, uses default title
    """
    # Import statements
    import matplotlib.pyplot as plt
    from network import NetworkClass
    from windstorm import WindClass
    from network_factory import make_network
    from windstorm_factory import make_windstorm

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

    # Bus data for labels
    bus_lon = net._get_bus_lon()
    bus_lat = net._get_bus_lat()
    bus_ids = net.data.net.bus

    # Validate branch data
    if not bch_gis_bgn or not bch_gis_end:
        print("Error: Branch GIS data is missing or invalid.")
        return

    # Check if branch_level and bus_level exist (moved here so they're always defined)
    has_branch_levels = hasattr(net.data.net, 'branch_level')
    has_bus_levels = hasattr(net.data.net, 'bus_level')

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot windstorm contours
    ax.scatter(start_lon, start_lat, color='blue', label='Starting-point Contour', zorder=2)
    for connection in start_concty:
        idx1, idx2 = connection[0] - 1, connection[1] - 1
        ax.plot([start_lon[idx1], start_lon[idx2]], [start_lat[idx1], start_lat[idx2]], 'b-', alpha=0.7, zorder=1)

    ax.scatter(end_lon, end_lat, color='red', label='Ending-point Contour', zorder=2)
    ax.plot(end_lon, end_lat, 'r-', alpha=0.7, zorder=1)

    # Plot branches with different colors for transmission and distribution levels
    # Create flags for legend (to avoid duplicate entries)
    tn_plotted = False
    dn_plotted = False

    for idx, (bgn, end) in enumerate(zip(bch_gis_bgn, bch_gis_end)):
        # Determine branch level if available
        if has_branch_levels:
            branch_level = net.data.net.branch_level.get(idx + 1, 'T')  # Default to 'T' if not found
        else:
            branch_level = 'T'  # Default to transmission if no level info

        # Set color and linewidth based on branch level
        if branch_level == 'T' or branch_level == 'T-D':
            color = 'darkgreen'
            linewidth = tn_linewidth
            label = 'Transmission Branch' if not tn_plotted else ""
            tn_plotted = True
        else:  # 'D'
            color = 'orange'
            linewidth = dn_linewidth
            label = 'Distribution Branch' if not dn_plotted else ""
            dn_plotted = True

        ax.plot([bgn[0], end[0]], [bgn[1], end[1]], color=color,
                alpha=0.8, linewidth=linewidth, zorder=1, label=label)

    # Draw bus markers & labels
    if label_buses:
        for lon, lat, bid in zip(bus_lon, bus_lat, bus_ids):
            # Determine bus level
            if has_bus_levels:
                bus_level = net.data.net.bus_level.get(bid, 'T')
            else:
                bus_level = 'T'

            # Set marker color based on bus level
            marker_color = 'darkgreen' if bus_level == 'T' else 'orange'

            # a tiny marker
            ax.scatter(lon, lat, s=15, c=marker_color, zorder=3)
            # text label, slightly offset in longitude
            ax.text(lon + label_offset_lon, lat,
                    str(bid),
                    fontsize=label_fontsize,
                    ha="left", va="center",
                    color=label_color,
                    zorder=4)

    # Set axis limits based on zoom option
    if zoomed_distribution and has_bus_levels:
        # Find distribution bus coordinates
        dn_lons = [lon for lon, bid in zip(bus_lon, bus_ids)
                   if net.data.net.bus_level.get(bid, 'T') == 'D']
        dn_lats = [lat for lat, bid in zip(bus_lat, bus_ids)
                   if net.data.net.bus_level.get(bid, 'T') == 'D']

        if dn_lons and dn_lats:
            # Calculate bounds with border
            min_lon = min(dn_lons) - zoom_border
            max_lon = max(dn_lons) + zoom_border
            min_lat = min(dn_lats) - zoom_border
            max_lat = max(dn_lats) + zoom_border

            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
        else:
            print("Warning: No distribution buses found for zooming.")
            ax.relim()
            ax.autoscale()
    else:
        # Auto scale for full view
        ax.relim()
        ax.autoscale()

    # Labels, title, and legend
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    # Set title - use custom title if provided, otherwise use default
    if title:
        ax.set_title(title,
                     fontsize=14,
                     # fontweight='bold'
                     )
    else:
        # Default title based on view mode
        if zoomed_distribution:
            ax.set_title(f"Branches and Windstorm Contours - {network_name} (Distribution Focus)",
                         fontsize=14,
                         # fontweight='bold'
                         )
        else:
            ax.set_title(f"Branches and Windstorm Contours - {network_name}",
                         fontsize=14,
                         # fontweight='bold'
                         )

    ax.legend()
    ax.grid(True)

    plt.show()


def visualize_bch_hrdn(results_xlsx: str,
                       plot_area: tuple | None = None,
                       zoomed_distribution: bool = False,
                       zoom_border: float = 0.1,
                       base_color: str = "lightgrey",
                       base_alpha: float = 0.3,
                       cmap_name: str = "YlOrRd",
                       line_width_scale: bool = True,
                       min_linewidth: float = 1.0,
                       max_linewidth: float = 4.0,
                       colorbar_label: str = "Hardening Level (m/s)",
                       colorbar_limits: tuple | str = "auto",
                       save_path: str = None,
                       title: str = None,
                       show_stats: bool = True,
                       # Font size parameters
                       title_fontsize: int = 14,
                       label_fontsize: int = 12,
                       tick_fontsize: int = 10,
                       stats_fontsize: int = 10,
                       colorbar_label_fontsize: int = 12,
                       colorbar_tick_fontsize: int = 10):
    """
    Visualize line hardening values from optimization results using a color scale.

    Parameters:
    -----------
    results_xlsx : str
        Path to the Excel file from write_selected_variables_to_excel
    plot_area : tuple or None
        (lon_min, lon_max, lat_min, lat_max) to zoom to specific area
    zoomed_distribution : bool
        If True, zoom in to show distribution network details
    zoom_border : float
        Border size for zoomed view (in degrees)
    base_color : str
        Color for non-hardened lines
    base_alpha : float
        Transparency for non-hardened lines
    cmap_name : str
        Matplotlib colormap name for hardening values
        Good options: 'YlOrRd', 'plasma', 'viridis', 'coolwarm', 'hot'
    line_width_scale : bool
        If True, scale line width based on hardening amount
    min_linewidth : float
        Minimum line width
    max_linewidth : float
        Maximum line width for maximum hardening
    colorbar_label : str
        Label for the colorbar
    colorbar_limits : tuple or str
        (vmin, vmax) for colorbar scale.
        If "auto", uses data min/max.
        If "network", uses limits from network configuration.
    save_path : str or None
        If provided, saves the figure
    title : str or None
        Custom title for the plot
    show_stats : bool
        If True, show statistics box on the plot
    title_fontsize : int
        Font size for the plot title
    label_fontsize : int
        Font size for axis labels
    tick_fontsize : int
        Font size for axis tick labels
    stats_fontsize : int
        Font size for statistics box
    colorbar_label_fontsize : int
        Font size for colorbar label
    colorbar_tick_fontsize : int
        Font size for colorbar tick labels

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    import pandas as pd
    import numpy as np

    # Read the Excel file
    book = pd.ExcelFile(results_xlsx)

    # Get metadata and load network
    meta = book.parse("Meta", header=None).set_index(0)[1].to_dict()
    net_name = meta.get("network_name", "default")
    net = NetworkClass() if net_name == "default" else make_network(net_name)
    net.set_gis_data()
    bgn = net._get_bch_gis_bgn()
    end = net._get_bch_gis_end()
    n_lines = len(bgn)

    # Get bus data for zoom functionality
    bus_lon = net._get_bus_lon()
    bus_lat = net._get_bus_lat()
    bus_ids = net.data.net.bus

    # Check if branch/bus levels exist
    has_branch_levels = hasattr(net.data.net, 'branch_level')
    has_bus_levels = hasattr(net.data.net, 'bus_level')

    # Read hardening values
    if "line_hrdn" not in book.sheet_names:
        print("Warning: Sheet 'line_hrdn' not found! Showing base network only.")
        hrdn_values = {}
    else:
        hrdn_df = book.parse("line_hrdn")
        # Parse hardening values into a dictionary
        hrdn_values = {}
        for idx, val in zip(hrdn_df["index"], hrdn_df["value"]):
            # Handle both string and integer index formats
            if isinstance(idx, str):
                # Index is a string like "(23,)" - extract the number
                branch_id = int(idx.strip("() ").split(",")[0])
            else:
                # Index is already an integer
                branch_id = int(idx)

            hrdn_values[branch_id] = float(val)

    # Get hardening statistics
    non_zero_values = [v for v in hrdn_values.values() if v > 0]
    if non_zero_values:
        min_hrdn = min(non_zero_values)
        max_hrdn = max(non_zero_values)
        avg_hrdn = sum(non_zero_values) / len(non_zero_values)
        total_hrdn = sum(non_zero_values)
    else:
        min_hrdn = max_hrdn = avg_hrdn = total_hrdn = 0

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Set up colormap and normalization
    cmap = cm.get_cmap(cmap_name)

    # Determine colorbar limits
    if colorbar_limits == "auto":
        vmin = 0
        vmax = max_hrdn if max_hrdn > 0 else 1
    elif colorbar_limits == "network":
        # Get limits directly from the network object we already loaded
        if hasattr(net.data, 'bch_hrdn_limits'):
            vmin, vmax = net.data.bch_hrdn_limits
            print(f"Using network hardening limits: {vmin} - {vmax} m/s")
        else:
            print("Warning: bch_hrdn_limits not found in network configuration. Using data limits.")
            vmin = 0
            vmax = max_hrdn if max_hrdn > 0 else 1
    else:
        # Manual limits provided as tuple
        vmin, vmax = colorbar_limits

    # Create normalization with the determined limits
    norm = Normalize(vmin=vmin, vmax=vmax)

    # First, draw all non-hardened branches with base color
    for l in range(1, n_lines + 1):
        if l not in hrdn_values or hrdn_values[l] == 0:
            p, q = bgn[l - 1], end[l - 1]

            # Optionally use different base colors for transmission/distribution
            if has_branch_levels:
                branch_level = net.data.net.branch_level.get(l, 'T')
                if branch_level == 'D':
                    color = 'lightblue'
                else:
                    color = base_color
            else:
                color = base_color

            ax.plot([p[0], q[0]], [p[1], q[1]],
                    color=color, alpha=base_alpha, linewidth=min_linewidth,
                    zorder=1)

    # Draw hardened lines with color scale
    for l, hrdn_val in hrdn_values.items():
        if hrdn_val > 0 and l <= n_lines:
            p, q = bgn[l - 1], end[l - 1]

            # Get color based on hardening value
            color = cmap(norm(hrdn_val))

            # Calculate line width if scaling is enabled
            if line_width_scale and vmax > vmin:
                # Linear interpolation between min and max width
                width_ratio = (hrdn_val - vmin) / (vmax - vmin)
                linewidth = min_linewidth + (max_linewidth - min_linewidth) * width_ratio
            else:
                linewidth = 2.5

            ax.plot([p[0], q[0]], [p[1], q[1]],
                    color=color, linewidth=linewidth,
                    solid_capstyle='round', zorder=2)

    # Add colorbar with customized font sizes
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label(colorbar_label, fontsize=colorbar_label_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    # Force colorbar limits
    cbar.mappable.set_clim(vmin, vmax)

    # Add ticks at reasonable intervals based on the scale
    if vmax <= 10:
        tick_interval = 1
    elif vmax <= 30:
        tick_interval = 5
    elif vmax <= 50:
        tick_interval = 10
    else:
        tick_interval = 20

    # Ensure ticks cover the full range
    ticks = np.arange(vmin, vmax + tick_interval / 2, tick_interval)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{int(t)}' for t in ticks])

    # Add statistics box with customized font size
    if show_stats:
        stats_text = f'Hardening Statistics:\n'
        if non_zero_values:
            stats_text += f'Lines hardened: {len(non_zero_values)}\n'
            stats_text += f'Total hardening: {total_hrdn:.1f} m/s\n'
            stats_text += f'Average: {avg_hrdn:.1f} m/s\n'
            stats_text += f'Range: {min_hrdn:.1f} - {max_hrdn:.1f} m/s'
        else:
            stats_text += 'No lines hardened'

        # # Add colorbar limits info
        # stats_text += f'\n\nColorbar limits: {vmin:.0f} - {vmax:.0f} m/s'
        # if colorbar_limits == "network":
        #     stats_text += ' (network)'
        # elif colorbar_limits == "auto":
        #     stats_text += ' (auto)'
        # else:
        #     stats_text += ' (manual)'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=stats_fontsize, verticalalignment='top', bbox=props)

    # Set axis limits based on zoom options
    if plot_area:
        # Manual plot area specification
        xmin, xmax, ymin, ymax = plot_area
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    elif zoomed_distribution and has_bus_levels:
        # Auto-zoom to distribution network
        dn_lons = [lon for lon, bid in zip(bus_lon, bus_ids)
                   if net.data.net.bus_level.get(bid, 'T') == 'D']
        dn_lats = [lat for lat, bid in zip(bus_lat, bus_ids)
                   if net.data.net.bus_level.get(bid, 'T') == 'D']

        if dn_lons and dn_lats:
            # Calculate bounds with border
            min_lon = min(dn_lons) - zoom_border
            max_lon = max(dn_lons) + zoom_border
            min_lat = min(dn_lats) - zoom_border
            max_lat = max(dn_lats) + zoom_border

            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
        else:
            print("Warning: No distribution buses found for zooming.")
            ax.relim()
            ax.autoscale()
            ax.margins(0.05)
    else:
        # Auto-scale with padding
        ax.relim()
        ax.autoscale()
        ax.margins(0.05)

    # Labels and title with customized font sizes
    ax.set_xlabel("Longitude", fontsize=label_fontsize)
    ax.set_ylabel("Latitude", fontsize=label_fontsize)

    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    else:
        title_text = f"Line Hardening Visualization - {net_name}"
        if zoomed_distribution:
            title_text += " (Distribution Focus)"
        ax.set_title(title_text, fontsize=title_fontsize, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    return fig, ax


def visualize_investment_vs_resilience(excel_file='Optimization_Results/Investment_Model/RM_expected_total_EENS_dn/4_ws_scenario_[112, 152, 166, 198]/data_for_plot.xlsx',
                                       figure_size=(10, 6),
                                       save_path=None,
                                       show_threshold=True,
                                       title=None):
    """
    Visualize the relationship between total investment cost and resilience metric (EENS).

    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing the data
    figure_size : tuple
        Size of the figure (width, height)
    save_path : str or None
        If provided, saves the figure to this path
    show_threshold : bool
        If True, shows both threshold and actual EENS values
    title : str or None
        Custom title for the plot. If None, uses default title

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Read the data
    df = pd.read_excel(excel_file)

    # Handle 'inf' values - convert to numeric, treating 'inf' as NaN initially
    df['resilience_metric_threshold'] = pd.to_numeric(df['resilience_metric_threshold'], errors='coerce')

    # Sort by investment cost for better line plotting
    df = df.sort_values('total_investment_cost')

    # Convert investment cost to millions for better readability
    df['investment_millions'] = df['total_investment_cost'] / 1e6

    # Create the figure
    fig, ax = plt.subplots(figsize=figure_size)

    # Plot the actual EENS vs investment cost
    ax.plot(df['investment_millions'], df['ws_exp_total_eens_dn'],
            'o-', color='darkblue', linewidth=2, markersize=8,
            label='Actual EENS at Distribution Level')

    # Optionally plot the threshold line
    if show_threshold:
        # Filter out rows where threshold is NaN (was 'inf')
        df_with_threshold = df[df['resilience_metric_threshold'].notna()]
        if not df_with_threshold.empty:
            ax.plot(df_with_threshold['investment_millions'],
                    df_with_threshold['resilience_metric_threshold'],
                    's--', color='red', linewidth=1.5, markersize=6, alpha=0.7,
                    label='EENS Threshold')

    # Add data point labels
    for idx, row in df.iterrows():
        # Show investment cost on each point
        ax.annotate(f'{row["investment_millions"]:.1f}M',
                    (row['investment_millions'], row['ws_exp_total_eens_dn']),
                    textcoords="offset points", xytext=(0, 10), ha='center',
                    fontsize=8, alpha=0.7)

    # Highlight the unconstrained case if it exists
    unconstrained = df[df['resilience_metric_threshold'].isna()]
    if not unconstrained.empty:
        ax.scatter(unconstrained['investment_millions'],
                   unconstrained['ws_exp_total_eens_dn'],
                   s=200, color='green', marker='*', zorder=5,
                   label='Unconstrained (No EENS limit)', edgecolors='black', linewidth=1)

    # Set labels and title
    ax.set_xlabel('Total Investment Cost (Million £)', fontsize=12)
    ax.set_ylabel('Expected Energy Not Supplied - EENS (MWh)', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Investment Cost vs Resilience Metric (EENS) Trade-off',
                     fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    ax.legend(loc='best', frameon=True, shadow=True)

    # Set y-axis to start from a reasonable minimum
    y_min = df['ws_exp_total_eens_dn'].min() * 0.95
    y_max = df['ws_exp_total_eens_dn'].max() * 1.05
    ax.set_ylim(y_min, y_max)

    # Format the axes
    ax.ticklabel_format(style='plain', axis='y')

    # Tight layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    return fig, ax


def visualize_investment_vs_resilience(excel_file='data_for_plot.xlsx',
                                       figure_size=(10, 6),
                                       save_path=None,
                                       show_threshold=True,
                                       title=None):
    """
    Visualize the relationship between total investment cost and resilience metric (EENS).

    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing the data
    figure_size : tuple
        Size of the figure (width, height)
    save_path : str or None
        If provided, saves the figure to this path
    show_threshold : bool
        If True, shows both threshold and actual EENS values
    title : str or None
        Custom title for the plot. If None, uses default title

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Read the data
    df = pd.read_excel(excel_file)

    # Handle 'inf' values - convert to numeric, treating 'inf' as NaN initially
    df['resilience_metric_threshold'] = pd.to_numeric(df['resilience_metric_threshold'], errors='coerce')

    # Sort by investment cost for better line plotting
    df = df.sort_values('total_investment_cost')

    # Convert investment cost to millions for better readability
    df['investment_millions'] = df['total_investment_cost'] / 1e6

    # Convert EENS to MWh (assuming it's already in MWh based on the values)
    # If it's in kWh, divide by 1000

    # Create the figure
    fig, ax = plt.subplots(figsize=figure_size)

    # Plot the actual EENS vs investment cost
    ax.plot(df['investment_millions'], df['ws_exp_total_eens_dn'],
            'o-', color='darkblue', linewidth=2, markersize=8,
            label='Actual EENS at Distribution Level')

    # Optionally plot the threshold line
    if show_threshold:
        # Filter out rows where threshold is NaN (was 'inf')
        df_with_threshold = df[df['resilience_metric_threshold'].notna()]
        if not df_with_threshold.empty:
            ax.plot(df_with_threshold['investment_millions'],
                    df_with_threshold['resilience_metric_threshold'],
                    's--', color='red', linewidth=1.5, markersize=6, alpha=0.7,
                    label='EENS Threshold')

    # Add data point labels
    for idx, row in df.iterrows():
        # Show investment cost on each point
        ax.annotate(f'{row["investment_millions"]:.1f}M',
                    (row['investment_millions'], row['ws_exp_total_eens_dn']),
                    textcoords="offset points", xytext=(0, 10), ha='center',
                    fontsize=8, alpha=0.7)

    # Highlight the unconstrained case if it exists
    unconstrained = df[df['resilience_metric_threshold'].isna() | np.isinf(df['resilience_metric_threshold'])]
    if not unconstrained.empty:
        ax.scatter(unconstrained['investment_millions'],
                   unconstrained['ws_exp_total_eens_dn'],
                   s=200, color='green', marker='*', zorder=5,
                   label='Unconstrained (No EENS limit)', edgecolors='black', linewidth=1)

    # Set labels and title
    ax.set_xlabel('Total Investment Cost (Million £)', fontsize=12)
    ax.set_ylabel('Expected Energy Not Supplied - EENS (MWh)', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Investment Cost vs Resilience Metric (EENS) Trade-off',
                     fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    ax.legend(loc='best', frameon=True, shadow=True)

    # Set y-axis to start from a reasonable minimum
    y_min = df['ws_exp_total_eens_dn'].min() * 0.95
    y_max = df['ws_exp_total_eens_dn'].max() * 1.05
    ax.set_ylim(y_min, y_max)

    # Format the axes
    ax.ticklabel_format(style='plain', axis='y')

    # Tight layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    return fig, ax


def visualize_investment_pareto_front(excel_file='data_for_plot.xlsx',
                                      figure_size=(10, 6),
                                      save_path=None,
                                      annotate_points=True,
                                      # New customization parameters
                                      marker_size=10,
                                      marker_color='darkgreen',
                                      line_color='darkgreen',
                                      line_width=2.5,
                                      show_feasible_region=True,  # New parameter
                                      feasible_color='lightgreen',
                                      feasible_alpha=0.2,
                                      show_stats=True,
                                      stats_position=(0.05, 0.95),
                                      title=None,
                                      title_fontsize=14,
                                      label_fontsize=12,
                                      tick_fontsize=10,
                                      annotation_fontsize=9,
                                      stats_fontsize=10,
                                      show_extreme_points=False,
                                      annotation_label="EENS Threshold (GWh)"):
    """
    Visualize the Pareto front of investment cost vs resilience metric,
    highlighting the trade-off curve.

    Parameters:
    -----------
    excel_file : str
        Path to the Excel file containing the data
    figure_size : tuple
        Size of the figure (width, height)
    save_path : str or None
        If provided, saves the figure to this path
    annotate_points : bool
        If True, annotates each point with its threshold value
    marker_size : int
        Size of the data point markers
    marker_color : str
        Color of the data points
    line_color : str
        Color of the Pareto front line
    line_width : float
        Width of the Pareto front line
    show_feasible_region : bool
        If True, shows the feasible region as a shaded area
    feasible_color : str
        Color of the feasible region
    feasible_alpha : float
        Transparency of the feasible region
    show_stats : bool
        If True, shows the statistics box
    stats_position : tuple
        Position of stats box (x, y) in figure coordinates
    title : str or None
        Custom title for the plot
    title_fontsize : int
        Font size for the title
    label_fontsize : int
        Font size for axis labels
    tick_fontsize : int
        Font size for tick labels
    annotation_fontsize : int
        Font size for point annotations
    stats_fontsize : int
        Font size for statistics box
    show_extreme_points : bool
        If True, highlights extreme points with special markers (legacy behavior)
    annotation_label : str
        Label to explain what the point annotations represent

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    # Read the data
    df = pd.read_excel(excel_file)

    # Handle 'inf' values
    df['resilience_metric_threshold'] = pd.to_numeric(df['resilience_metric_threshold'], errors='coerce')
    # Replace inf with NaN for easier handling
    df['resilience_metric_threshold'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Sort by EENS (descending) to create proper Pareto front
    df = df.sort_values('ws_exp_total_eens_dn', ascending=False)

    # Convert to millions/thousands for readability
    df['investment_millions'] = df['total_investment_cost'] / 1e6
    df['eens_thousands'] = df['ws_exp_total_eens_dn'] / 1000

    # Create the figure
    fig, ax = plt.subplots(figsize=figure_size)

    # Plot the Pareto front
    ax.plot(df['eens_thousands'], df['investment_millions'],
            'o-', color=line_color, linewidth=line_width, markersize=marker_size,
            markeredgecolor='black', markeredgewidth=1,
            label='Pareto Front')  # Add label for the line

    # Fill area under the curve to show feasible region (optional)
    if show_feasible_region:
        ax.fill_between(df['eens_thousands'], df['investment_millions'],
                        df['investment_millions'].max() * 1.1,
                        alpha=feasible_alpha, color=feasible_color,
                        label='Feasible Region')

    # Annotate points with threshold values if requested
    if annotate_points:
        for idx, row in df.iterrows():
            if pd.isna(row['resilience_metric_threshold']) or np.isinf(row['resilience_metric_threshold']):
                label = 'No limit'
            else:
                # Convert from MWh to GWh for display
                threshold_gwh = row['resilience_metric_threshold'] / 1000
                # Format with appropriate precision
                if threshold_gwh >= 1:
                    label = f"{threshold_gwh:.1f}"
                else:
                    label = f"{threshold_gwh:.2f}"

            # Determine annotation position based on curve direction
            if idx == 0:
                xytext = (15, -5)
            elif idx == len(df) - 1:
                xytext = (-15, 5)
            else:
                xytext = (10, 10)

            ax.annotate(label,
                        (row['eens_thousands'], row['investment_millions']),
                        textcoords="offset points", xytext=xytext,
                        fontsize=annotation_fontsize,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="gray", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2",
                                        color="gray", alpha=0.6))

    # Optionally highlight extreme points (legacy behavior)
    if show_extreme_points:
        # Minimum investment (unconstrained)
        min_inv = df.loc[df['investment_millions'].idxmin()]
        ax.scatter(min_inv['eens_thousands'], min_inv['investment_millions'],
                   s=200, color='red', marker='s', zorder=5,
                   label='Minimum Investment', edgecolors='darkred', linewidth=2)

        # Minimum EENS (highest investment)
        min_eens = df.loc[df['eens_thousands'].idxmin()]
        ax.scatter(min_eens['eens_thousands'], min_eens['investment_millions'],
                   s=200, color='blue', marker='^', zorder=5,
                   label='Minimum EENS', edgecolors='darkblue', linewidth=2)

    # Set labels and title
    ax.set_xlabel('EENS at distribution level across all windstorm scenarios (GWh)', fontsize=label_fontsize)
    ax.set_ylabel('Total investment cost (Million £)', fontsize=label_fontsize)

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    else:
        ax.set_title('Pareto Front: Investment Cost vs Resilience Metric Trade-off',
                     fontsize=title_fontsize, fontweight='bold')

    # Set tick font sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Get existing handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Add a dummy handle for annotation explanation if annotations are shown
    if annotate_points:
        # Create invisible line with annotation box as marker
        annotation_handle = Line2D([0], [0], marker='s', color='w',
                                   markerfacecolor='white', markeredgecolor='gray',
                                   markersize=8, markeredgewidth=1,
                                   label=f'Labels show {annotation_label}')
        handles.append(annotation_handle)
        labels.append(f'Labels show {annotation_label}')

    # Add legend with all handles
    ax.legend(handles=handles, labels=labels, loc='upper right',
              frameon=True, shadow=True, fontsize=annotation_fontsize)

    # Add text box with key insights (optional)
    if show_stats:
        # Update stats to show GWh
        textstr = f'Investment Range: £{df["investment_millions"].min():.1f}M - £{df["investment_millions"].max():.1f}M\n'
        textstr += f'EENS Range: {df["eens_thousands"].min():.1f} - {df["eens_thousands"].max():.1f} GWh'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(stats_position[0], stats_position[1], textstr, transform=ax.transAxes,
                fontsize=stats_fontsize, verticalalignment='top', bbox=props)

    # Tight layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    return fig, ax