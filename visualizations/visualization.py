# This script contains function for visualization

from core.network import NetworkClass
from core.windstorm import WindClass
from factories.network_factory import make_network

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import lognorm
import json
import math

from factories.windstorm_factory import make_windstorm



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


def visualize_fragility_curve_shift(WindConfig,
                                    hardening_levels=[10, 20, 30],
                                    colors=['blue', 'green', 'orange', 'red'],
                                    show_arrow=True,
                                    show_textbox=True,
                                    title="Fragility Curve Shift due to Line Hardening",
                                    save_path=None,
                                    figsize=(10, 7),
                                    # Font size parameters
                                    title_fontsize=14,
                                    axis_label_fontsize=12,
                                    axis_tick_fontsize=10,
                                    legend_fontsize=10,
                                    arrow_text_fontsize=11,
                                    textbox_fontsize=10,
                                    threshold_label_fontsize=9):
    """
    Visualize how line hardening shifts the fragility curve to the right.

    Parameters:
    -----------
    WindConfig : WindConfig object
        The wind configuration containing fragility parameters
    hardening_levels : list of float
        List of hardening amounts (m/s) to show. Default: [10, 20, 30]
    colors : list of str
        Colors for each curve (original + hardened curves)
    show_arrow : bool
        If True, shows an arrow indicating the shift direction
    show_textbox : bool
        If True, shows the explanatory text box. Default: True
    title : str
        Plot title
    save_path : str or None
        If provided, saves the figure
    figsize : tuple
        Figure size as (width, height). Default: (10, 7)
    title_fontsize : int
        Font size for the plot title. Default: 14
    axis_label_fontsize : int
        Font size for x and y axis labels. Default: 12
    axis_tick_fontsize : int
        Font size for axis tick labels. Default: 10
    legend_fontsize : int
        Font size for legend entries. Default: 10
    arrow_text_fontsize : int
        Font size for the arrow annotation text. Default: 11
    textbox_fontsize : int
        Font size for the explanatory text box. Default: 10
    threshold_label_fontsize : int
        Font size for threshold line labels in legend. Default: 9

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import lognorm
    from matplotlib.patches import FancyArrowPatch

    wcon = WindConfig

    # Get parameters from config
    mu = wcon.data.frg.mu
    sigma = wcon.data.frg.sigma
    thrd_1 = wcon.data.frg.thrd_1
    thrd_2 = wcon.data.frg.thrd_2
    shift_f = wcon.data.frg.shift_f

    # Generate hazard intensities
    hzd_int_range = np.linspace(0, 150, 500)  # Extended range to show shifted curves

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot original fragility curve
    pof_values_original = []
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
        pof_values_original.append(pof)

    ax.plot(hzd_int_range, pof_values_original,
            color=colors[0], linewidth=2.5, label='Original Fragility Curve')

    # Plot shifted fragility curves for each hardening level
    for i, hardening in enumerate(hardening_levels, 1):
        pof_values_shifted = []
        for hzd_int in hzd_int_range:
            # Hardening shifts the curve to the right
            f_hzd_int = hzd_int - shift_f - hardening
            if f_hzd_int < thrd_1:
                pof = 0
            elif f_hzd_int > thrd_2:
                pof = 1
            else:
                shape = sigma
                scale = np.exp(mu)
                pof = lognorm.cdf(f_hzd_int, s=shape, scale=scale)
            pof_values_shifted.append(pof)

        # Use dashed line for hardened curves
        ax.plot(hzd_int_range, pof_values_shifted,
                color=colors[i % len(colors)], linewidth=2, linestyle='--',
                label=f'Hardened (+{hardening} m/s)')

    # Add threshold vertical lines for original curve
    ax.axvline(thrd_1 + shift_f, color='gray', linestyle=':', alpha=0.5,
               label=f'Threshold 1 ({thrd_1 + shift_f})')
    ax.axvline(thrd_2 + shift_f, color='gray', linestyle=':', alpha=0.5,
               label=f'Threshold 2 ({thrd_2 + shift_f})')

    # Add arrow to show shift if requested
    if show_arrow and len(hardening_levels) > 0:
        # Find the x-coordinate where the original curve reaches PoF ≈ 0.5
        target_pof = 0.5
        # Find the index closest to 0.5 probability
        idx_05 = np.argmin(np.abs(np.array(pof_values_original) - target_pof))
        x_original = hzd_int_range[idx_05]
        x_hardened = x_original + hardening_levels[0]

        # Get the actual PoF value at this x-coordinate
        actual_pof = pof_values_original[idx_05]

        # Create curved arrow
        arrow = FancyArrowPatch((x_original, actual_pof), (x_hardened, actual_pof),
                                connectionstyle="arc3,rad=0",
                                arrowstyle='->',
                                mutation_scale=25,
                                linewidth=2,
                                color='black',
                                alpha=0.7)
        ax.add_patch(arrow)

        # Add text annotation slightly above the arrow
        ax.text((x_original + x_hardened) / 2, actual_pof + 0.05,
                'Hardening\nShift',
                ha='center', va='bottom',
                fontsize=arrow_text_fontsize, fontweight='bold')

    # Labels and formatting with customizable font sizes
    ax.set_xlabel('Wind Speed (m/s)', fontsize=axis_label_fontsize)
    ax.set_ylabel('Failure Probability', fontsize=axis_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(120, thrd_2 + shift_f + max(hardening_levels) + 10))

    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=axis_tick_fontsize)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend with customizable font size
    ax.legend(loc='center right', frameon=True, shadow=True, fontsize=legend_fontsize)

    # Add annotation box explaining the concept (only if show_textbox is True)
    if show_textbox:
        textstr = 'Line hardening shifts the fragility\ncurve rightward, requiring higher\nwind speeds to cause failure'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.7, textstr, transform=ax.transAxes, fontsize=textbox_fontsize,
                verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return fig, ax


def visualize_fragility_curve_comparison(WindConfig,
                                         hardening_amount=20,
                                         figure_size=(12, 6),
                                         save_path=None):
    """
    Side-by-side comparison of original and hardened fragility curves.

    Parameters:
    -----------
    WindConfig : WindConfig object
        The wind configuration containing fragility parameters
    hardening_amount : float
        Amount of hardening shift in m/s
    figure_size : tuple
        Figure size (width, height)
    save_path : str or None
        If provided, saves the figure

    Returns:
    --------
    fig, (ax1, ax2) : matplotlib figure and axis objects
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import lognorm

    wcon = WindConfig

    # Get parameters from config
    mu = wcon.data.frg.mu
    sigma = wcon.data.frg.sigma
    thrd_1 = wcon.data.frg.thrd_1
    thrd_2 = wcon.data.frg.thrd_2
    shift_f = wcon.data.frg.shift_f

    # Generate hazard intensities
    hzd_int_range = np.linspace(0, 120, 500)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size, sharey=True)

    # Calculate fragility for original curve
    pof_original = []
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
        pof_original.append(pof)

    # Calculate fragility for hardened curve
    pof_hardened = []
    for hzd_int in hzd_int_range:
        f_hzd_int = hzd_int - shift_f - hardening_amount
        if f_hzd_int < thrd_1:
            pof = 0
        elif f_hzd_int > thrd_2:
            pof = 1
        else:
            shape = sigma
            scale = np.exp(mu)
            pof = lognorm.cdf(f_hzd_int, s=shape, scale=scale)
        pof_hardened.append(pof)

    # Plot original curve
    ax1.plot(hzd_int_range, pof_original, 'b-', linewidth=2.5, label='Fragility Curve')
    ax1.axvline(thrd_1 + shift_f, color='green', linestyle='--',
                label=f'Threshold 1 ({thrd_1 + shift_f})')
    ax1.axvline(thrd_2 + shift_f, color='red', linestyle='--',
                label=f'Threshold 2 ({thrd_2 + shift_f})')
    ax1.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax1.set_ylabel('Failure Probability', fontsize=12)
    ax1.set_title('Original Line', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_ylim(-0.05, 1.05)

    # Plot hardened curve
    ax2.plot(hzd_int_range, pof_hardened, 'r-', linewidth=2.5, label='Hardened Fragility Curve')
    ax2.axvline(thrd_1 + shift_f + hardening_amount, color='green', linestyle='--',
                label=f'Threshold 1 ({thrd_1 + shift_f + hardening_amount})')
    ax2.axvline(thrd_2 + shift_f + hardening_amount, color='red', linestyle='--',
                label=f'Threshold 2 ({thrd_2 + shift_f + hardening_amount})')
    ax2.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax2.set_title(f'Hardened Line (+{hardening_amount} m/s)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    # Add main title
    fig.suptitle('Effect of Line Hardening on Fragility Curve', fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return fig, (ax1, ax2)


def visualize_windstorm_event(file_path, scenario_number, event_number,
                              custom_title=None,
                              # Font size parameters
                              title_fontsize=14,
                              xlabel_fontsize=12,
                              ylabel_fontsize=12,
                              tick_fontsize=10,
                              legend_fontsize=10,
                              # Legend parameters
                              legend_loc="best",
                              legend_bbox_to_anchor=None,
                              legend_ncol=1):
    """
    Visualizes the path of a windstorm event from the stored ws_scenarios .json files.

    Parameters:
    - file_path: Path to the JSON file containing windstorm scenarios
    - scenario_number: Scenario number (1-based)
    - event_number: Event number (1-based)
    - custom_title: Custom title for the plot. If None, uses default format
    - title_fontsize: Font size for the plot title
    - xlabel_fontsize: Font size for x-axis label
    - ylabel_fontsize: Font size for y-axis label
    - tick_fontsize: Font size for axis tick labels
    - legend_fontsize: Font size for legend
    - legend_loc: Location of legend
    - legend_bbox_to_anchor: Tuple (x, y) for custom legend position
    - legend_ncol: Number of columns for legend entries
    """
    # Import necessary modules
    import json
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from factories.network_factory import make_network
    from factories.windstorm_factory import make_windstorm

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
    bch_gis_bgn = net._get_bch_gis_bgn()
    bch_gis_end = net._get_bch_gis_end()

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
    ax.set_xlabel("Longitude", fontsize=xlabel_fontsize)
    ax.set_ylabel("Latitude", fontsize=ylabel_fontsize)

    # Set tick font sizes
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    # Use custom title if provided, otherwise use default
    if custom_title:
        ax.set_title(custom_title, fontsize=title_fontsize, fontweight='bold')
    else:
        ax.set_title(f"Windstorm Path - Scenario {scenario_number}, Event {event_number}",
                     fontsize=title_fontsize, fontweight='bold')

    # Set legend with custom font size and position
    if legend_bbox_to_anchor:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc,
                  bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol)
    else:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc, ncol=legend_ncol)

    ax.grid(True)

    plt.show()


def visualize_all_windstorm_events(
        file_path="Scenario_Database/all_full_scenarios_year.json"
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
                                 title: str = None,
                                 show_windstorm_contours: bool = True,
                                 # Font size parameters
                                 title_fontsize: int = 14,
                                 xlabel_fontsize: int = 12,
                                 ylabel_fontsize: int = 12,
                                 tick_fontsize: int = 10,
                                 legend_fontsize: int = 10,
                                 # Legend parameters
                                 legend_loc: str = "best",
                                 legend_bbox_to_anchor: tuple = None,
                                 legend_ncol: int = 1):
    """
    Visualize the branches along with the starting- and ending-points contour for windstorm path generation.
    Now includes bus ID labels similar to visualize_network_bch.

    Parameters:
    - network_name: Name of the network preset registered in network_factory
    - windstorm_name: Name of the windstorm preset registered in windstorm_factory
    - label_buses: If True, write the bus ID next to each plotted node
    - label_fontsize: Font size for the bus ID labels
    - label_offset_lon: Horizontal offset (in degrees) applied to the label
    - label_color: Text colour of the bus labels
    - zoomed_distribution: If True, zoom in to show distribution network details
    - zoom_border: Border size for zoomed view (in degrees)
    - tn_linewidth: Line thickness for transmission branches
    - dn_linewidth: Line thickness for distribution branches
    - title: Custom title for the plot. If None, uses default title
    - show_windstorm_contours: If True, shows windstorm contours. If False, shows only network branches
    - title_fontsize: Font size for the plot title
    - xlabel_fontsize: Font size for x-axis label
    - ylabel_fontsize: Font size for y-axis label
    - tick_fontsize: Font size for axis tick labels
    - legend_fontsize: Font size for legend
    - legend_loc: Location of legend. Options: 'best', 'upper right', 'upper left', 'lower left',
                  'lower right', 'right', 'center left', 'center right', 'lower center',
                  'upper center', 'center'
    - legend_bbox_to_anchor: Tuple (x, y) for custom legend position. E.g., (1.05, 1) places legend
                             outside plot area on the right
    - legend_ncol: Number of columns for legend entries
    """
    # Import statements
    import matplotlib.pyplot as plt
    from core.network import NetworkClass
    from core.windstorm import WindClass
    from factories.network_factory import make_network
    from factories.windstorm_factory import make_windstorm

    # load network model
    if network_name == 'default':
        net = NetworkClass()
    else:
        net = make_network(network_name)

    # load windstorm model only if needed
    if show_windstorm_contours:
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

    # Plot windstorm contours only if requested
    if show_windstorm_contours:
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
            # branch_level is a dict with 1-based keys
            branch_level = net.data.net.branch_level.get(idx + 1, 'T')
            if branch_level == 'T' or branch_level == 'TN':
                color = 'darkgreen'
                alpha = 0.8
                lw = tn_linewidth
                label = 'Transmission Branches' if not tn_plotted else None
                tn_plotted = True
            elif branch_level == 'D' or branch_level == 'DN':
                color = 'orange'
                alpha = 0.6
                lw = dn_linewidth
                label = 'Distribution Branches' if not dn_plotted else None
                dn_plotted = True
            else:  # 'T-D' coupling branch
                color = 'purple'
                alpha = 0.7
                lw = (tn_linewidth + dn_linewidth) / 2
                label = None
        else:
            # Default if branch_level doesn't exist
            color = 'darkgreen'
            alpha = 0.8
            lw = 1.5
            label = 'Network Branches' if idx == 0 else None

        ax.plot([bgn[0], end[0]], [bgn[1], end[1]],
                color=color, alpha=alpha, lw=lw, label=label, zorder=3)

    # Plot buses with labels if requested
    if label_buses:
        # Different colors for transmission and distribution buses if level info exists
        if has_bus_levels:
            for lon, lat, bid in zip(bus_lon, bus_lat, bus_ids):
                # bus_level is a dict indexed by bus ID
                bus_level = net.data.net.bus_level.get(bid, 'T')

                if bus_level == 'T' or bus_level == 'TN':
                    marker_color = 'darkgreen'
                    marker_size = 30
                else:  # 'D' or 'DN'
                    marker_color = 'orange'
                    marker_size = 20

                # Plot marker
                ax.scatter(lon, lat, s=marker_size, c=marker_color, zorder=4)
                # Plot label
                ax.text(lon + label_offset_lon, lat, str(bid),
                        fontsize=label_fontsize, ha="left", va="center",
                        color=label_color, zorder=5)
        else:
            # Default if bus_level doesn't exist
            for lon, lat, bid in zip(bus_lon, bus_lat, bus_ids):
                ax.scatter(lon, lat, s=20, c='black', zorder=4)
                ax.text(lon + label_offset_lon, lat, str(bid),
                        fontsize=label_fontsize, ha="left", va="center",
                        color=label_color, zorder=5)

    # Handle zooming
    if zoomed_distribution and has_bus_levels:
        # Find distribution bus bounds
        dn_lons = []
        dn_lats = []
        for i, bid in enumerate(bus_ids):
            # bus_level is a dict indexed by bus ID
            bus_level = net.data.net.bus_level.get(bid, 'T')
            if bus_level == 'D' or bus_level == 'DN':
                dn_lons.append(bus_lon[i])
                dn_lats.append(bus_lat[i])

        if dn_lons:  # Only zoom if distribution buses exist
            min_lon, max_lon = min(dn_lons), max(dn_lons)
            min_lat, max_lat = min(dn_lats), max(dn_lats)

            # Add border
            lon_range = max_lon - min_lon
            lat_range = max_lat - min_lat
            min_lon -= zoom_border * lon_range
            max_lon += zoom_border * lon_range
            min_lat -= zoom_border * lat_range
            max_lat += zoom_border * lat_range

            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
        else:
            # If no distribution buses found, auto scale
            ax.relim()
            ax.autoscale()
    else:
        # Auto scale for full view
        ax.relim()
        ax.autoscale()

    # Labels, title, and legend
    ax.set_xlabel("Longitude", fontsize=xlabel_fontsize)
    ax.set_ylabel("Latitude", fontsize=ylabel_fontsize)

    # Set tick font sizes
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    # Set title - use custom title if provided, otherwise use default
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold')
    else:
        # Default title based on view mode and content
        if show_windstorm_contours:
            if zoomed_distribution:
                ax.set_title(f"Branches and Windstorm Contours - {network_name} (Distribution Focus)",
                             fontsize=title_fontsize, fontweight='bold')
            else:
                ax.set_title(f"Branches and Windstorm Contours - {network_name}",
                             fontsize=title_fontsize, fontweight='bold')
        else:
            if zoomed_distribution:
                ax.set_title(f"Network Branches - {network_name} (Distribution Focus)",
                             fontsize=title_fontsize, fontweight='bold')
            else:
                ax.set_title(f"Network Branches - {network_name}",
                             fontsize=title_fontsize, fontweight='bold')

    # Set legend with custom font size and position
    if legend_bbox_to_anchor:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc,
                  bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol)
    else:
        ax.legend(fontsize=legend_fontsize, loc=legend_loc, ncol=legend_ncol)
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


def visualize_investment_pareto_front(excel_file=None,
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
                label = 'infinite'
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
                                   label=f'{annotation_label}')
        handles.append(annotation_handle)
        labels.append(f'{annotation_label}')

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


def visualize_scenario_tree(tree,
                            title=None,
                            figure_size=(12, 8),
                            save_path=None,
                            show_probabilities=True,
                            show_years=True,
                            node_size=1000,
                            font_size=10,
                            edge_width=2,
                            show_stats=True,
                            layout='hierarchical'):
    """
    Visualize a scenario tree structure.

    Parameters:
    -----------
    tree : ScenarioTree
        The scenario tree object to visualize
    title : str or None
        Custom title for the plot. If None, uses default
    figure_size : tuple
        Size of the figure (width, height)
    save_path : str or None
        If provided, saves the figure to this path
    show_probabilities : bool
        If True, shows transition probabilities on edges
    show_years : bool
        If True, shows years in node labels
    node_size : int
        Size of the nodes in the visualization
    font_size : int
        Font size for labels
    edge_width : float
        Width of the edges
    show_stats : bool
        If True, shows statistics box
    layout : str
        Layout algorithm: 'hierarchical' or 'spring'

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)

    # Create networkx graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for node_id, node in tree.nodes.items():
        G.add_node(node_id,
                   stage=node.stage,
                   year=node.year,
                   prob=node.cumulative_probability)

    # Add edges with transition probabilities
    edge_probs = {}
    for node_id, node in tree.nodes.items():
        if node.parent_id is not None:
            G.add_edge(node.parent_id, node_id)
            edge_probs[(node.parent_id, node_id)] = node.transition_probability

    # Create layout
    if layout == 'hierarchical':
        # Hierarchical layout for tree structure
        pos = {}
        stage_counts = {}
        stage_current = {}

        # Count nodes per stage
        for node in tree.nodes.values():
            stage_counts[node.stage] = stage_counts.get(node.stage, 0) + 1
            stage_current[node.stage] = 0

        # Position nodes
        for node_id, node in tree.nodes.items():
            stage = node.stage
            x = stage * 3  # Horizontal spacing between stages

            # Vertical positioning
            total_in_stage = stage_counts[stage]
            current_index = stage_current[stage]
            y = (current_index - (total_in_stage - 1) / 2) * 1.5

            pos[node_id] = (x, y)
            stage_current[stage] += 1
    else:
        # Spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)

    # Determine node colors
    node_colors = []
    for node_id in G.nodes():
        node = tree.nodes[node_id]
        if node.is_root():
            node_colors.append('lightgreen')
        elif node.is_leaf():
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightblue')

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_size,
                           alpha=0.9,
                           ax=ax)

    # Draw edges
    nx.draw_networkx_edges(G, pos,
                           edge_color='gray',
                           width=edge_width,
                           alpha=0.6,
                           arrows=True,
                           arrowsize=20,
                           ax=ax)

    # Add node labels
    labels = {}
    for node_id, node in tree.nodes.items():
        label_parts = []

        if node.is_root():
            label_parts.append("Root")
        else:
            label_parts.append(f"N{node_id}")

        if show_years:
            label_parts.append(f"{node.year}")

        labels[node_id] = "\n".join(label_parts)

    nx.draw_networkx_labels(G, pos, labels,
                            font_size=font_size,
                            ax=ax)

    # Add edge labels (probabilities)
    if show_probabilities and edge_probs:
        edge_labels = {edge: f"{prob:.2f}" for edge, prob in edge_probs.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                     font_size=font_size * 0.8,
                                     ax=ax)

    # Add statistics box
    if show_stats:
        stats_text = f"Scenario Tree Statistics\n"
        stats_text += f"━━━━━━━━━━━━━━━━━━━━\n"
        stats_text += f"Total nodes: {len(tree.nodes)}\n"
        stats_text += f"Stages: {len(tree.stages)}\n"
        stats_text += f"Years: {tree.stages}\n"
        stats_text += f"Scenarios: {tree.get_num_scenarios()}\n"

        # Check probability sum
        leaf_prob_sum = sum(node.cumulative_probability
                            for node in tree.nodes.values()
                            if node.is_leaf())
        stats_text += f"Prob. sum: {leaf_prob_sum:.4f}"

        # Add warning if probabilities don't sum to 1
        if abs(leaf_prob_sum - 1.0) > 0.001:
            stats_text += " ⚠️"

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=font_size * 0.9,
                verticalalignment='top',
                bbox=props)

    # Set title
    if title:
        ax.set_title(title, fontsize=font_size * 1.4, fontweight='bold')
    else:
        ax.set_title("Scenario Tree Structure", fontsize=font_size * 1.4, fontweight='bold')

    # Remove axes
    ax.axis('off')

    # Adjust layout
    if layout == 'hierarchical':
        # Set reasonable bounds for hierarchical layout
        x_margin = 1
        y_margin = 1
        x_min = -x_margin
        x_max = (len(tree.stages) - 1) * 3 + x_margin

        all_y = [p[1] for p in pos.values()]
        y_min = min(all_y) - y_margin
        y_max = max(all_y) + y_margin

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    return fig, ax


def visualize_scenario_tree_compact(tree,
                                    title=None,
                                    figure_size=(10, 6),
                                    save_path=None,
                                    show_legend=True):
    """
    Compact visualization of scenario tree showing probability distribution.

    Parameters:
    -----------
    tree : ScenarioTree
        The scenario tree object to visualize
    title : str or None
        Custom title for the plot
    figure_size : tuple
        Size of the figure (width, height)
    save_path : str or None
        If provided, saves the figure to this path
    show_legend : bool
        If True, shows legend

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)

    # Get leaf nodes and their properties
    scenarios = []
    probabilities = []
    paths = []

    for node_id, node in tree.nodes.items():
        if node.is_leaf():
            # Get path from root to leaf
            path = []
            current = node
            while current is not None:
                path.append(current.node_id)
                current = tree.nodes.get(current.parent_id)
            path.reverse()

            scenarios.append(f"Scenario {len(scenarios) + 1}")
            probabilities.append(node.cumulative_probability)
            paths.append(path)

    # Create bar plot
    x = np.arange(len(scenarios))
    bars = ax.bar(x, probabilities, color='skyblue', edgecolor='navy', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=10)

    # Customize plot
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45 if len(scenarios) > 6 else 0, ha='right')

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(probabilities) * 1.15)

    # Add title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"Scenario Probability Distribution ({len(scenarios)} scenarios)",
                     fontsize=14, fontweight='bold')

    # Add total probability check
    total_prob = sum(probabilities)
    ax.text(0.98, 0.02, f'Total: {total_prob:.4f}',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax