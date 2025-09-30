"""
Network topology visualization utilities.

This module provides comprehensive network topology plotting capabilities
with flexible customization options including zoom levels, windstorm overlays,
and detailed formatting control.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from typing import Optional, Tuple, List, Dict, Any

from factories.network_factory import make_network
from factories.windstorm_factory import make_windstorm


# -----------------------------
# Helper Functions
# -----------------------------
def plot_network_branches(ax,
                         net,
                         tn_color: str = '#1f77b4',
                         dn_color: str = '#ff7f0e',
                         lw: float = 1.5,
                         alpha: float = 0.8,
                         show_legend: bool = True):
    """
    Plot network branches with transmission/distribution distinction.

    Parameters
    ----------
    ax : matplotlib axes
        Axes to plot on
    net : NetworkClass
        Network object with GIS data
    tn_color : str
        Color for transmission branches
    dn_color : str
        Color for distribution branches
    lw : float
        Line width for branches
    alpha : float
        Transparency for branches
    show_legend : bool
        Whether to add legend entries
    """
    # Get branch coordinates
    net.set_gis_data()
    bgn = net._get_bch_gis_bgn()
    end = net._get_bch_gis_end()

    has_branch_levels = hasattr(net.data.net, "branch_level")

    tn_plotted = False
    dn_plotted = False

    for idx, (p, q) in enumerate(zip(bgn, end), start=1):
        if has_branch_levels:
            lvl = net.data.net.branch_level.get(idx, "T")
        else:
            lvl = "T"

        if lvl in ("T", "T-D"):
            color = tn_color
            label = "Transmission branch" if not tn_plotted and show_legend else None
            tn_plotted = True
        else:
            color = dn_color
            label = "Distribution branch" if not dn_plotted and show_legend else None
            dn_plotted = True

        ax.plot([p[0], q[0]], [p[1], q[1]],
                color=color, lw=lw, alpha=alpha, label=label)


def plot_network_buses(ax, net,
                       tn_node_color='#2ca02c',
                       dn_node_color='#d62728',
                       node_size=30,
                       alpha=1.0,
                       show_legend=True,
                       label_buses=False,
                       label_fontsize=8,
                       label_offset=0.02):
    """
    Plot network buses on given axes.

    Parameters
    ----------
    ax : matplotlib axes
        Axes to plot on
    net : NetworkClass
        Network object with GIS data
    tn_node_color : str
        Color for transmission buses
    dn_node_color : str
        Color for distribution buses
    node_size : float
        Size of bus markers
    alpha : float
        Transparency for buses
    show_legend : bool
        Whether to add legend entries
    label_buses : bool
        Whether to label bus numbers
    label_fontsize : int
        Font size for bus labels
    label_offset : float
        Offset for bus labels (in degrees)
    """
    has_bus_levels = hasattr(net.data.net, "bus_level")

    tn_buses_plotted = False
    dn_buses_plotted = False

    for i, bus_id in enumerate(net.data.net.bus):
        lon = net.data.net.bus_lon[i]
        lat = net.data.net.bus_lat[i]

        if has_bus_levels:
            lvl = net.data.net.bus_level.get(bus_id, "T")
        else:
            lvl = "T"

        if lvl == "T":
            color = tn_node_color
            label = "Transmission Bus" if not tn_buses_plotted and show_legend else None
            tn_buses_plotted = True
        else:
            color = dn_node_color
            label = "Distribution Bus" if not dn_buses_plotted and show_legend else None
            dn_buses_plotted = True

        ax.scatter(lon, lat, c=color, s=node_size, alpha=alpha,
                   zorder=3, label=label)

        # Add bus labels if requested
        if label_buses:
            # Add white background box to make text more readable
            text = ax.text(lon + label_offset, lat, str(bus_id),
                           fontsize=label_fontsize,
                           ha='left',
                           va='center',
                           weight='bold',  # Make text bolder for clarity
                           zorder=5,  # Ensure text is on top
                           bbox=dict(boxstyle='round,pad=0.2',
                                     facecolor='white',
                                     edgecolor='gray',
                                     alpha=0.8,
                                     linewidth=0.5))


def add_windstorm_contours(ax,
                          ws,
                          event_data: Dict[str, Any],
                          start_color: str = 'green',
                          end_color: str = 'red',
                          contour_alpha: float = 0.3,
                          show_legend: bool = True):
    """
    Add windstorm starting and ending point contours to the plot.

    Parameters
    ----------
    ax : matplotlib axes
        Axes to plot on
    ws : WindClass
        Windstorm object
    event_data : dict
        Event data containing epicentres and radius information
    start_color : str
        Color for starting contour
    end_color : str
        Color for ending contour
    contour_alpha : float
        Transparency for contours
    show_legend : bool
        Whether to add legend entries
    """
    epicentres = np.array(event_data["epicentres"])
    radius_km = np.array(event_data["radius_km"])

    if len(epicentres) == 0:
        return

    # Convert radius from km to degrees (approximate)
    lat_avg = np.mean(epicentres[:, 1])
    r_deg = radius_km / 111.0  # Approximate conversion

    # Plot starting point contour
    start_circle = Circle(
        (epicentres[0, 0], epicentres[0, 1]),
        r_deg[0],
        facecolor=start_color,
        edgecolor=start_color,
        alpha=contour_alpha,
        label="Windstorm Start" if show_legend else None
    )
    ax.add_patch(start_circle)

    # Plot ending point contour
    end_circle = Circle(
        (epicentres[-1, 0], epicentres[-1, 1]),
        r_deg[-1],
        facecolor=end_color,
        edgecolor=end_color,
        alpha=contour_alpha,
        label="Windstorm End" if show_legend else None
    )
    ax.add_patch(end_circle)


def set_network_view_bounds(ax,
                           net,
                           zoom_to_dn: bool = False,
                           padding_factor: float = 0.1):
    """
    Set the view bounds for the network plot.

    Parameters
    ----------
    ax : matplotlib axes
        Axes to plot on
    net : NetworkClass
        Network object
    zoom_to_dn : bool
        If True, zoom to distribution network area only
    padding_factor : float
        Padding factor for the bounds (as fraction of range)
    """
    if zoom_to_dn and hasattr(net.data.net, "bus_level"):
        # Get only distribution bus coordinates
        dn_lons = []
        dn_lats = []
        for i, bus_id in enumerate(net.data.net.bus):
            if net.data.net.bus_level.get(bus_id, "T") == "D":
                dn_lons.append(net.data.net.bus_lon[i])
                dn_lats.append(net.data.net.bus_lat[i])

        if dn_lons:  # If we have distribution buses
            bus_lons = dn_lons
            bus_lats = dn_lats
        else:
            # Fallback to all buses if no distribution buses found
            bus_lons = net.data.net.bus_lon
            bus_lats = net.data.net.bus_lat
    else:
        # Use all buses
        bus_lons = net.data.net.bus_lon
        bus_lats = net.data.net.bus_lat

    # Calculate bounds with padding
    lon_range = max(bus_lons) - min(bus_lons)
    lat_range = max(bus_lats) - min(bus_lats)

    lon_padding = lon_range * padding_factor
    lat_padding = lat_range * padding_factor

    ax.set_xlim(min(bus_lons) - lon_padding, max(bus_lons) + lon_padding)
    ax.set_ylim(min(bus_lats) - lat_padding, max(bus_lats) + lat_padding)


# -----------------------------
# Main Visualization Function
# -----------------------------
def visualize_network_topology(
    network_preset: str = "29_bus_GB_transmission_network_with_Kearsley_GSP_group",
    windstorm_preset: Optional[str] = None,
    windstorm_event_path: Optional[str] = None,
    scenario_number: Optional[int] = None,
    event_number: Optional[int] = None,
    zoom_to_dn: bool = False,
    show_dn_inset: bool = False,
    inset_position: str = 'upper right',
    inset_size: float = 0.3,
    inset_border_color: str = 'black',
    inset_border_width: float = 1.5,
    inset_borderpad: float = 3.0,
    inset_lw_scale: float = 0.8,
    inset_bus_scale: float = 0.7,
    inset_padding_scale: float = 0.5,
    inset_show_title: bool = True,
    inset_title: str = "DN Detail",
    inset_title_fontsize: int = 8,
    show_windstorm_contours: bool = False,
    show_buses: bool = True,
    label_buses: bool = False,
    interconnection_buses: Optional[List[int]] = None,
    show_interconnection_arrows: bool = False,
    ic_arrow_length: float = 0.3,
    ic_arrow_color: str = 'purple',
    ic_arrow_width: float = 2.5,
    ic_arrow_style: str = 'simple',
    ic_arrow_directions: Optional[Dict[int, float]] = None,
    custom_title: Optional[str] = None,
    title_fontsize: int = 14,
    xlabel_fontsize: int = 12,
    ylabel_fontsize: int = 12,
    tick_fontsize: int = 10,
    legend_fontsize: int = 10,
    legend_loc: str = 'best',
    legend_ncol: int = 1,
    figsize: Tuple[int, int] = (12, 10),
    tn_branch_color: str = '#1f77b4',
    dn_branch_color: str = '#ff7f0e',
    tn_bus_color: str = '#2ca02c',
    dn_bus_color: str = '#d62728',
    branch_lw: float = 1.5,
    branch_alpha: float = 0.8,
    bus_size: float = 30,
    bus_alpha: float = 1.0,
    label_fontsize: int = 8,
    label_offset: float = 0.02,
    ws_start_color: str = 'green',
    ws_end_color: str = 'red',
    ws_contour_alpha: float = 0.3,
    padding_factor: float = 0.1,
    show_grid: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    figure_margins: Optional[Dict[str, float]] = None,
    tight_layout_pad: float = 1.08,
    save_pad_inches: float = 0.1
):
    """
    Visualize network topology with flexible customization options.

    Parameters
    ----------
    network_preset : str
        Network preset name from network_factory
    windstorm_preset : str, optional
        Windstorm preset name for contour plotting
    windstorm_event_path : str, optional
        Path to windstorm event JSON file for contour data
    scenario_number : int, optional
        Scenario number (1-based) if using windstorm data
    event_number : int, optional
        Event number (1-based) if using windstorm data
    zoom_to_dn : bool
        If True, zoom to distribution network area only
    show_dn_inset : bool
        If True, show zoomed DN view as inset on full network plot
    inset_position : str
        Position of inset ('upper right', 'upper left', 'lower right', 'lower left')
    inset_size : float
        Size of inset as fraction of main plot (0.2-0.4 recommended)
    inset_border_color : str
        Color of inset border
    inset_border_width : float
        Width of inset border
    inset_borderpad : float
        Gap between inset and main plot edges in points (default 3.0)
        Larger values push the inset further from edges
    inset_lw_scale : float
        Scale factor for line width in inset relative to main plot (default 0.8)
    inset_bus_scale : float
        Scale factor for bus marker size in inset relative to main plot (default 0.7)
    inset_padding_scale : float
        Scale factor for padding in inset relative to main plot (default 0.5)
    inset_show_title : bool
        Whether to show title in inset (default True)
    inset_title : str
        Title text for inset (default "DN Detail")
    inset_title_fontsize : int
        Font size for inset title (default 8)
    show_windstorm_contours : bool
        If True, show windstorm start/end contours
    show_buses : bool
        If True, plot network buses
    label_buses : bool
        If True, add bus number labels
    interconnection_buses : list of int, optional
        List of bus IDs that are grid interconnection points
    show_interconnection_arrows : bool
        If True, show arrows at interconnection points
    ic_arrow_length : float
        Length of interconnection arrows in coordinate units
    ic_arrow_color : str
        Color of interconnection arrows
    ic_arrow_width : float
        Width of interconnection arrow lines
    ic_arrow_style : str
        Style of arrows ('simple', 'fancy', 'double')
    ic_arrow_directions : dict, optional
        Custom directions for arrows {bus_id: angle_degrees}
    custom_title : str, optional
        Custom title for the plot
    title_fontsize : int
        Font size for title
    xlabel_fontsize : int
        Font size for x-axis label
    ylabel_fontsize : int
        Font size for y-axis label
    tick_fontsize : int
        Font size for tick labels
    legend_fontsize : int
        Font size for legend
    legend_loc : str
        Legend location
    legend_ncol : int
        Number of columns in legend
    figsize : tuple
        Figure size (width, height)
    tn_branch_color : str
        Color for transmission branches
    dn_branch_color : str
        Color for distribution branches
    tn_bus_color : str
        Color for transmission buses
    dn_bus_color : str
        Color for distribution buses
    branch_lw : float
        Line width for branches
    branch_alpha : float
        Transparency for branches
    bus_size : float
        Size of bus markers
    bus_alpha : float
        Transparency for buses
    label_fontsize : int
        Font size for bus labels
    label_offset : float
        Offset for bus labels (in degrees)
    ws_start_color : str
        Color for windstorm start contour
    ws_end_color : str
        Color for windstorm end contour
    ws_contour_alpha : float
        Transparency for windstorm contours
    padding_factor : float
        Padding factor for view bounds
    show_grid : bool
        Whether to show grid
    save_path : str, optional
        Path to save the figure
    dpi : int
        DPI for saved figure
    figure_margins : dict, optional
        Manual figure margins: {'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.1}
        Values are fractions of figure width/height
    tight_layout_pad : float
        Padding for tight_layout in inches (default 1.08)
        Only used when show_dn_inset=False
    save_pad_inches : float
        Padding around figure when saving in inches (default 0.1)
        Controls white space around the saved image

    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Load network
    net = make_network(network_preset)

    # Plot branches
    plot_network_branches(
        ax, net,
        tn_color=tn_branch_color,
        dn_color=dn_branch_color,
        lw=branch_lw,
        alpha=branch_alpha,
        show_legend=True
    )

    # Plot buses if requested
    if show_buses:
        plot_network_buses(
            ax, net,
            tn_node_color=tn_bus_color,
            dn_node_color=dn_bus_color,
            node_size=bus_size,
            alpha=bus_alpha,
            show_legend=True,
            label_buses=label_buses,
            label_fontsize=label_fontsize,
            label_offset=label_offset
        )

    # Add windstorm contours if requested
    if show_windstorm_contours and windstorm_event_path:
        try:
            # Load windstorm data
            with open(windstorm_event_path, 'r') as f:
                ws_data = json.load(f)

            # Get windstorm preset
            if windstorm_preset is None:
                windstorm_preset = ws_data.get('metadata', {}).get('windstorm_preset', 'windstorm_GB_transmission_network')

            ws = make_windstorm(windstorm_preset)

            # Get scenarios
            if "scenarios" in ws_data:
                scenarios = ws_data["scenarios"]
            elif "ws_scenarios" in ws_data:
                scenarios = ws_data["ws_scenarios"]
            else:
                print("No windstorm scenarios found in data")
                scenarios = []

            # Get specific event if provided
            if scenario_number and event_number and scenarios:
                scenario_idx = scenario_number - 1
                if scenario_idx < len(scenarios):
                    scenario = scenarios[scenario_idx]
                    events = scenario.get("events", [])
                    event_idx = event_number - 1

                    if event_idx < len(events):
                        event_data = events[event_idx]
                        add_windstorm_contours(
                            ax, ws, event_data,
                            start_color=ws_start_color,
                            end_color=ws_end_color,
                            contour_alpha=ws_contour_alpha,
                            show_legend=True
                        )
                    else:
                        print(f"Event {event_number} not found in scenario {scenario_number}")
                else:
                    print(f"Scenario {scenario_number} not found")
        except Exception as e:
            print(f"Could not load windstorm data: {e}")

    # Set view bounds
    set_network_view_bounds(ax, net, zoom_to_dn=zoom_to_dn, padding_factor=padding_factor)

    # Add interconnection arrows if requested
    if show_interconnection_arrows and interconnection_buses:
        from matplotlib.patches import FancyArrowPatch

        # Get bus coordinates
        bus_coords = {}
        for i, bus_id in enumerate(net.data.net.bus):
            bus_coords[bus_id] = (net.data.net.bus_lon[i], net.data.net.bus_lat[i])

        # Calculate center of network for automatic arrow directions
        if ic_arrow_directions is None:
            center_lon = np.mean([coord[0] for coord in bus_coords.values()])
            center_lat = np.mean([coord[1] for coord in bus_coords.values()])
            ic_arrow_directions = {}

            for bus_id in interconnection_buses:
                if bus_id in bus_coords:
                    bus_lon, bus_lat = bus_coords[bus_id]
                    # Calculate angle from center to bus (outward direction)
                    angle = np.arctan2(bus_lat - center_lat, bus_lon - center_lon)
                    ic_arrow_directions[bus_id] = np.degrees(angle)

        # Add arrows for each interconnection point
        for bus_id in interconnection_buses:
            if bus_id not in bus_coords:
                print(f"Warning: Interconnection bus {bus_id} not found in network")
                continue

            bus_lon, bus_lat = bus_coords[bus_id]
            angle_deg = ic_arrow_directions.get(bus_id, 0)
            angle_rad = np.radians(angle_deg)

            # Calculate arrow end point
            end_lon = bus_lon + ic_arrow_length * np.cos(angle_rad)
            end_lat = bus_lat + ic_arrow_length * np.sin(angle_rad)

            if ic_arrow_style == 'simple':
                # Simple arrow
                ax.annotate('',
                           xy=(end_lon, end_lat),
                           xytext=(bus_lon, bus_lat),
                           arrowprops=dict(
                               arrowstyle='->',
                               color=ic_arrow_color,
                               lw=ic_arrow_width,
                               shrinkA=5,
                               shrinkB=0
                           ))
            elif ic_arrow_style == 'fancy':
                # Fancy arrow
                arrow = FancyArrowPatch(
                    (bus_lon, bus_lat),
                    (end_lon, end_lat),
                    arrowstyle='->',
                    color=ic_arrow_color,
                    linewidth=ic_arrow_width,
                    mutation_scale=20,
                    connectionstyle="arc3,rad=0"
                )
                ax.add_patch(arrow)
            elif ic_arrow_style == 'double':
                # Double-headed arrow
                ax.annotate('',
                           xy=(end_lon, end_lat),
                           xytext=(bus_lon, bus_lat),
                           arrowprops=dict(
                               arrowstyle='<->',
                               color=ic_arrow_color,
                               lw=ic_arrow_width,
                               shrinkA=5,
                               shrinkB=0
                           ))

    # Add DN inset if requested
    if show_dn_inset and not zoom_to_dn and hasattr(net.data.net, "bus_level"):
        # Check if there are distribution buses
        dn_buses_exist = any(net.data.net.bus_level.get(bus_id, "T") == "D"
                             for bus_id in net.data.net.bus)

        if dn_buses_exist:
            print(f"Creating DN inset at position: {inset_position}")
            # Import required modules for inset
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            # Use absolute units (inches) for better compatibility
            # Map position strings to matplotlib location codes
            loc_map = {
                'upper right': 1,
                'upper left': 2,
                'lower left': 3,
                'lower right': 4
            }

            # Calculate size in inches based on figure size
            fig_width, fig_height = figsize
            inset_size_inches = min(fig_width, fig_height) * inset_size

            # Create inset with absolute size to avoid tight_layout issues
            ax_inset = inset_axes(ax,
                                  width=inset_size_inches,  # inches
                                  height=inset_size_inches,  # inches
                                  loc=loc_map.get(inset_position, 1),
                                  borderpad=inset_borderpad)

            # Plot branches in inset (without legend)
            plot_network_branches(
                ax_inset, net,
                tn_color=tn_branch_color,
                dn_color=dn_branch_color,
                lw=branch_lw * inset_lw_scale,  # Use customizable scale
                alpha=branch_alpha,
                show_legend=False
            )

            # Plot buses in inset if requested
            if show_buses:
                plot_network_buses(
                    ax_inset, net,
                    tn_node_color=tn_bus_color,
                    dn_node_color=dn_bus_color,
                    node_size=bus_size * inset_bus_scale,  # Use customizable scale
                    alpha=bus_alpha,
                    show_legend=False,
                    label_buses=False,  # No labels in inset
                    label_fontsize=label_fontsize,
                    label_offset=label_offset
                )

            # Set DN view bounds for inset
            set_network_view_bounds(ax_inset, net, zoom_to_dn=True,
                                   padding_factor=padding_factor * inset_padding_scale)

            # Style the inset
            ax_inset.set_xlabel("")
            ax_inset.set_ylabel("")
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])

            # Add border to inset
            for spine in ax_inset.spines.values():
                spine.set_edgecolor(inset_border_color)
                spine.set_linewidth(inset_border_width)

            # Optional: Add a small title to inset
            if inset_show_title:
                ax_inset.text(0.5, 0.95, inset_title,
                             transform=ax_inset.transAxes,
                             ha='center', va='top',
                             fontsize=inset_title_fontsize, fontweight='bold')

            # Add grid if requested
            if show_grid:
                ax_inset.grid(True, alpha=0.2, linewidth=0.5)

            print(f"DN inset created successfully at {inset_position}")
        else:
            print("Warning: No distribution buses found in network - DN inset not created")

    # Labels and title
    ax.set_xlabel("Longitude", fontsize=xlabel_fontsize)
    ax.set_ylabel("Latitude", fontsize=ylabel_fontsize)

    # Set tick font sizes
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    # Set title
    if custom_title:
        ax.set_title(custom_title, fontsize=title_fontsize, fontweight='bold')
    else:
        if zoom_to_dn:
            title = f"Network Topology - {network_preset} (Distribution View)"
        else:
            title = f"Network Topology - {network_preset}"
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold')

    # Legend
    handles, labels = ax.get_legend_handles_labels()

    # Add interconnection arrow to legend if shown
    if show_interconnection_arrows and interconnection_buses:
        from matplotlib.lines import Line2D
        # Create arrow marker for legend based on arrow style
        if ic_arrow_style == 'double':
            # Double arrow for legend
            ic_handle = Line2D([0], [0], marker=r'$\leftrightarrow$',
                              color=ic_arrow_color,
                              linewidth=0,  # No line, just marker
                              markersize=12,
                              markeredgewidth=1.5,
                              label='Grid interconnection')
        else:
            # Single arrow for legend
            ic_handle = Line2D([0], [0], marker='$\\rightarrow$',
                              color=ic_arrow_color,
                              linewidth=0,  # No line, just marker
                              markersize=12,
                              markeredgewidth=1.5,
                              label='Grid interconnection')
        handles.append(ic_handle)
        labels.append('Grid interconnection')

    ax.legend(handles=handles, labels=labels, fontsize=legend_fontsize,
              loc=legend_loc, ncol=legend_ncol)

    # Grid
    if show_grid:
        ax.grid(True, alpha=0.3)

    # Apply manual figure margins if specified
    if figure_margins:
        plt.subplots_adjust(
            left=figure_margins.get('left', 0.1),
            right=figure_margins.get('right', 0.9),
            top=figure_margins.get('top', 0.9),
            bottom=figure_margins.get('bottom', 0.1)
        )
    elif not show_dn_inset:
        # Only use tight_layout if not using inset and no manual margins
        plt.tight_layout(pad=tight_layout_pad)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=save_pad_inches)
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig, ax


# -----------------------------
# Additional Utility Functions
# -----------------------------
def compare_network_topologies(
    network_presets: List[str],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 6),
    **kwargs
):
    """
    Compare multiple network topologies side by side.

    Parameters
    ----------
    network_presets : list of str
        List of network preset names to compare
    titles : list of str, optional
        Custom titles for each subplot
    figsize : tuple
        Figure size (width, height)
    **kwargs
        Additional keyword arguments passed to visualize_network_topology

    Returns
    -------
    fig, axes : matplotlib figure and list of axes
    """
    n_networks = len(network_presets)
    fig, axes = plt.subplots(1, n_networks, figsize=figsize)

    if n_networks == 1:
        axes = [axes]

    for i, (preset, ax) in enumerate(zip(network_presets, axes)):
        # Load network
        net = make_network(preset)

        # Plot branches
        plot_network_branches(
            ax, net,
            tn_color=kwargs.get('tn_branch_color', '#1f77b4'),
            dn_color=kwargs.get('dn_branch_color', '#ff7f0e'),
            lw=kwargs.get('branch_lw', 1.5),
            alpha=kwargs.get('branch_alpha', 0.8),
            show_legend=(i == 0)  # Only show legend on first plot
        )

        # Plot buses if requested
        if kwargs.get('show_buses', True):
            plot_network_buses(
                ax, net,
                tn_node_color=kwargs.get('tn_bus_color', '#2ca02c'),
                dn_node_color=kwargs.get('dn_bus_color', '#d62728'),
                node_size=kwargs.get('bus_size', 30),
                alpha=kwargs.get('bus_alpha', 1.0),
                show_legend=(i == 0),
                label_buses=kwargs.get('label_buses', False),
                label_fontsize=kwargs.get('label_fontsize', 8),
                label_offset=kwargs.get('label_offset', 0.02)
            )

        # Set bounds
        set_network_view_bounds(
            ax, net,
            zoom_to_dn=kwargs.get('zoom_to_dn', False),
            padding_factor=kwargs.get('padding_factor', 0.1)
        )

        # Labels and title
        ax.set_xlabel("Longitude", fontsize=kwargs.get('xlabel_fontsize', 12))
        ax.set_ylabel("Latitude", fontsize=kwargs.get('ylabel_fontsize', 12))
        ax.tick_params(axis='both', labelsize=kwargs.get('tick_fontsize', 10))

        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=kwargs.get('title_fontsize', 14))
        else:
            ax.set_title(preset, fontsize=kwargs.get('title_fontsize', 14))

        if kwargs.get('show_grid', True):
            ax.grid(True, alpha=0.3)

        if i == 0 and kwargs.get('show_legend', True):
            ax.legend(fontsize=kwargs.get('legend_fontsize', 10),
                     loc=kwargs.get('legend_loc', 'best'))

    plt.tight_layout()

    if kwargs.get('save_path'):
        plt.savefig(kwargs['save_path'], dpi=kwargs.get('dpi', 300), bbox_inches='tight')

    plt.show()
    return fig, axes


# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    """
    Example usage of network topology visualization.
    """

    # Example 1: Basic network topology
    # print("Example 1: Basic network topology visualization")
    # visualize_network_topology(
    #     network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
    #     zoom_to_dn=True,
    #     show_buses=True,
    #     label_buses=False,
    #     custom_title="GB Transmission Network with Kearsley GSP Group"
    # )

    # Example 2: Zoomed distribution network view
    print("\nExample 2: Distribution network view")
    visualize_network_topology(
        network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
        zoom_to_dn=True,
        show_buses=True,
        label_buses=True,
        label_fontsize=14,
        label_offset=0.01,
        custom_title="Distribution Network - Kearsley GSP Group (Zoomed View)"
    )

    # Example 3: Network with windstorm contours
    # Note: This requires a valid windstorm event file
    # print("\nExample 3: Network with windstorm contours")
    # visualize_network_topology(
    #     network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
    #     windstorm_event_path="Scenario_Database/all_full_scenarios_year.json",
    #     scenario_number=1,
    #     event_number=1,
    #     show_windstorm_contours=True,
    #     custom_title="Network Topology with Windstorm Event"
    # )

    # Example 4: Customized visualization
    # print("\nExample 4: Customized network visualization")
    # visualize_network_topology(
    #     network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
    #     show_buses=False,
    #     label_buses=False,
    #     tn_branch_color='green',
    #     dn_branch_color='orange',
    #     tn_bus_color='darkgreen',
    #     dn_bus_color='darkorange',
    #     branch_lw=1.5,
    #     bus_size=50,
    #     title_fontsize=16,
    #     xlabel_fontsize=12,
    #     ylabel_fontsize=12,
    #     tick_fontsize=12,
    #     legend_fontsize=12,
    #     figsize=(8, 10),
    #     custom_title="Customized Network Visualization",
    #     show_grid=True
    # )

    # Example 5: Network with DN inset
    # print("\nExample 5: Full network with distribution network inset")
    # visualize_network_topology(
    #     network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
    #     show_dn_inset=True,
    #     inset_position='upper right',
    #     inset_size=0.35,
    #     show_buses=True,
    #     label_buses=False,
    #     custom_title="GB Network with DN Inset View",
    #     legend_loc='upper left',  # Move legend to avoid inset
    #     figsize=(12, 10)
    # )

    # Example 6: Compare different network views
    # print("\nExample 5: Network comparison")
    # compare_network_topologies(
    #     network_presets=["GB_transmission_network",
    #                      "29_bus_GB_transmission_network_with_Kearsley_GSP_group"],
    #     titles=["GB Transmission Only", "With Kearsley Distribution"],
    #     show_buses=True,
    #     figsize=(16, 6)
    # )

    # visualize_network_topology(
    #     network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
    #     show_buses=True,
    #     label_buses=False,
    #     tn_branch_color='green',
    #     dn_branch_color='orange',
    #     tn_bus_color='darkgreen',
    #     dn_bus_color='darkorange',
    #     branch_lw=1.5,
    #     bus_size=50,
    #     title_fontsize=16,
    #     xlabel_fontsize=14,
    #     ylabel_fontsize=14,
    #     tick_fontsize=12,
    #     legend_fontsize=14,
    #     legend_loc='center left',
    #     figsize=(10, 10),
    #     custom_title="Network Topology Visualisation",
    #     # Add interconnection arrows
    #     interconnection_buses=[5, 10, 11, 26, 27],  # Your IC buses
    #     show_interconnection_arrows=True,
    #     ic_arrow_color='purple',
    #     ic_arrow_width=3,
    #     ic_arrow_length=0.5,
    #     ic_arrow_directions={
    #         5: -135,  # Adjust arrow angles)
    #         10: 0,
    #         11: 180,
    #         26: 0,
    #         27: -45
    #     },
    #     # Inset
    #     show_dn_inset=True,
    #     inset_position='upper right',
    #     inset_size=0.30,
    #     inset_borderpad=1.5,  # Gap from edges (points)
    #     inset_lw_scale=0.8,  # Line width 80% of main
    #     inset_bus_scale=0.7,  # Bus size 70% of main
    #     # inset_padding_scale=0.5,  # Padding 50% of main
    #     inset_show_title=True,  # Show title in inset
    #     inset_title="DN: Kearsley GSP Group",  # Custom title text
    #     inset_title_fontsize=12,  # Title font size
    #     show_grid=True,
    #     figure_margins={  # Manual margins with inset
    #         'left': 0.08,
    #         'right': 0.94,
    #         'top': 0.94,
    #         'bottom': 0.08
    #     },
    # )