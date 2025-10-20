"""
Fragility curve visualization utilities.

This module provides comprehensive fragility curve plotting capabilities
with flexible customization options including parameter loading from network
presets (via net.data.frg), manual specification, and detailed formatting control.

Note: When loading from network presets, fragility parameters are accessed from
net.data.frg.{mu, sigma, thrd_1, thrd_2, shift_f}. If these are lists (one value
per branch), the first value is used for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from typing import Optional, Union, List, Tuple, Dict, Any
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Rectangle

# Try to import network factory, but make it optional
try:
    from factories.network_factory import make_network
    HAS_NETWORK_FACTORY = True
except ImportError:
    HAS_NETWORK_FACTORY = False
    make_network = None


# -----------------------------
# Helper Functions
# -----------------------------
def calculate_fragility_pof(wind_speeds: np.ndarray,
                           mu: float,
                           sigma: float,
                           thrd_1: float,
                           thrd_2: float,
                           shift_f: float,
                           hardening_shift: float = 0) -> np.ndarray:
    """
    Calculate probability of failure for given wind speeds.

    Parameters
    ----------
    wind_speeds : np.ndarray
        Array of wind speeds to calculate PoF for
    mu : float
        Logarithmic mean parameter of lognormal distribution
    sigma : float
        Logarithmic standard deviation parameter
    thrd_1 : float
        Lower threshold (PoF = 0 below this)
    thrd_2 : float
        Upper threshold (PoF = 1 above this)
    shift_f : float
        Base shift of fragility curve
    hardening_shift : float
        Additional shift due to hardening (rightward shift)

    Returns
    -------
    np.ndarray
        Array of failure probabilities
    """
    pof_values = []

    for wind_speed in wind_speeds:
        # Apply shifts (hardening shifts curve to the right)
        f_wind_speed = wind_speed - shift_f - hardening_shift

        if f_wind_speed < thrd_1:
            pof = 0
        elif f_wind_speed > thrd_2:
            pof = 1
        else:
            shape = sigma
            scale = np.exp(mu)
            pof = lognorm.cdf(f_wind_speed, s=shape, scale=scale)

        pof_values.append(pof)

    return np.array(pof_values)


def plot_single_fragility_curve(ax: plt.Axes,
                               wind_speeds: np.ndarray,
                               pof_values: np.ndarray,
                               thrd_1: float,
                               thrd_2: float,
                               shift_f: float,
                               hardening_shift: float = 0,
                               curve_color: str = 'blue',
                               curve_linewidth: float = 2.5,
                               curve_label: str = 'Fragility Curve',
                               show_thresholds: bool = True,
                               threshold_colors: List[str] = ['green', 'red'],
                               threshold_linestyles: List[str] = ['--', '--'],
                               threshold_linewidth: float = 1.5,
                               threshold_alpha: float = 0.7,
                               threshold_labels: bool = True,
                               threshold_fontsize: int = 9):
    """
    Plot a single fragility curve on given axes.

    Parameters
    ----------
    ax : matplotlib axes
        Axes to plot on
    wind_speeds : np.ndarray
        Wind speed values
    pof_values : np.ndarray
        Probability of failure values
    thrd_1, thrd_2 : float
        Threshold values
    shift_f : float
        Base shift amount
    hardening_shift : float
        Additional hardening shift
    curve_color : str
        Color of the main curve
    curve_linewidth : float
        Width of the curve line
    curve_label : str
        Label for the curve
    show_thresholds : bool
        Whether to show threshold lines
    threshold_colors : list
        Colors for threshold lines
    threshold_linestyles : list
        Styles for threshold lines
    threshold_linewidth : float
        Width of threshold lines
    threshold_alpha : float
        Transparency of threshold lines
    threshold_labels : bool
        Whether to add labels for thresholds
    threshold_fontsize : int
        Font size for threshold labels
    """
    # Plot main curve
    ax.plot(wind_speeds, pof_values,
           color=curve_color,
           linewidth=curve_linewidth,
           label=curve_label)

    # Plot threshold lines if requested
    if show_thresholds:
        actual_thrd_1 = thrd_1 + shift_f + hardening_shift
        actual_thrd_2 = thrd_2 + shift_f + hardening_shift

        # Always add threshold lines to legend with simple labels
        ax.axvline(actual_thrd_1,
                   color=threshold_colors[0],
                   linestyle=threshold_linestyles[0],
                   linewidth=threshold_linewidth,
                   alpha=threshold_alpha,
                   label='Lower threshold')

        ax.axvline(actual_thrd_2,
                   color=threshold_colors[1],
                   linestyle=threshold_linestyles[1],
                   linewidth=threshold_linewidth,
                   alpha=threshold_alpha,
                   label='Upper threshold')

        # Add text annotations if threshold_labels is True
        if threshold_labels:
            # Add text labels near the threshold lines
            ax_ylim = ax.get_ylim()
            y_pos = ax_ylim[0] + (ax_ylim[1] - ax_ylim[0]) * 0.95  # Position near top

            ax.text(actual_thrd_1, y_pos, f'{actual_thrd_1:.1f} m/s',
                    rotation=90, verticalalignment='top',
                    fontsize=threshold_fontsize, color=threshold_colors[0])

            ax.text(actual_thrd_2, y_pos, f'{actual_thrd_2:.1f} m/s',
                    rotation=90, verticalalignment='top',
                    fontsize=threshold_fontsize, color=threshold_colors[1])


def add_hardening_arrow(ax: plt.Axes,
                       original_wind: float,
                       hardened_wind: float,
                       pof_level: float,
                       arrow_color: str = 'black',
                       arrow_style: str = 'simple',
                       arrow_width: float = 2,
                       arrow_head_width: float = 0.03,
                       arrow_head_length: float = 2,
                       text_label: str = 'Hardening\nShift',
                       text_fontsize: int = 11,
                       text_fontweight: str = 'bold',
                       text_offset: float = 0.05):
    """
    Add an arrow showing the hardening shift direction.

    Parameters
    ----------
    ax : matplotlib axes
        Axes to plot on
    original_wind : float
        Wind speed at original curve
    hardened_wind : float
        Wind speed at hardened curve
    pof_level : float
        PoF level to place arrow at
    arrow_color : str
        Color of the arrow
    arrow_style : str
        Style of arrow ('simple', 'fancy', 'wedge')
    arrow_width : float
        Width of arrow shaft
    arrow_head_width : float
        Width of arrow head (as fraction of y-axis range for simple arrow)
    arrow_head_length : float
        Length of arrow head (in data units for simple arrow)
    text_label : str
        Text to display near arrow
    text_fontsize : int
        Font size for arrow text
    text_fontweight : str
        Font weight for arrow text
    text_offset : float
        Vertical offset for text above arrow
    """
    if arrow_style == 'fancy':
        # Use FancyArrowPatch for more control
        arrow = FancyArrowPatch((original_wind, pof_level),
                               (hardened_wind - 1, pof_level),  # Slightly shorter to make room for head
                               connectionstyle="arc3,rad=0",
                               arrowstyle=f'->',
                               mutation_scale=20,
                               color=arrow_color,
                               linewidth=arrow_width,
                               zorder=5)
        ax.add_patch(arrow)
    else:
        # Use simple arrow with better proportions
        # Draw horizontal line
        ax.plot([original_wind, hardened_wind - arrow_head_length],
                [pof_level, pof_level],
                color=arrow_color,
                linewidth=arrow_width,
                zorder=5)

        # Draw arrow head using annotate for better control
        ax.annotate('', xy=(hardened_wind, pof_level),
                   xytext=(hardened_wind - arrow_head_length, pof_level),
                   arrowprops=dict(arrowstyle='->',
                                 color=arrow_color,
                                 lw=arrow_width,
                                 shrinkA=0, shrinkB=0))

    # Add text label
    ax.text((original_wind + hardened_wind) / 2, pof_level + text_offset,
           text_label,
           ha='center', va='bottom',
           fontsize=text_fontsize,
           fontweight=text_fontweight)


def add_info_textbox(ax: plt.Axes,
                    text: str,
                    position: Union[str, Tuple[float, float]] = 'upper left',
                    fontsize: int = 10,
                    box_color: str = 'wheat',
                    box_alpha: float = 0.8,
                    box_style: str = 'round',
                    edge_color: str = 'black',
                    edge_width: float = 1):
    """
    Add an information text box to the plot.

    Parameters
    ----------
    ax : matplotlib axes
        Axes to add text box to
    text : str
        Text to display
    position : str or tuple
        Position of text box ('upper left', 'upper right', etc.) or (x, y) coordinates
    fontsize : int
        Font size for text
    box_color : str
        Background color of box
    box_alpha : float
        Transparency of box
    box_style : str
        Style of box corners
    edge_color : str
        Color of box edge
    edge_width : float
        Width of box edge
    """
    # Convert position strings to coordinates
    position_map = {
        'upper left': (0.02, 0.98),
        'upper right': (0.98, 0.98),
        'lower left': (0.02, 0.02),
        'lower right': (0.98, 0.02),
        'center': (0.5, 0.5),
        'upper center': (0.5, 0.98),
        'lower center': (0.5, 0.02),
        'center left': (0.02, 0.5),
        'center right': (0.98, 0.5)
    }

    if isinstance(position, str):
        x, y = position_map.get(position, (0.02, 0.98))
    else:
        x, y = position

    # Determine alignment based on position
    ha = 'left' if x < 0.5 else 'right' if x > 0.5 else 'center'
    va = 'bottom' if y < 0.5 else 'top' if y > 0.5 else 'center'

    props = dict(boxstyle=f'{box_style},pad=0.3',
                facecolor=box_color,
                alpha=box_alpha,
                edgecolor=edge_color,
                linewidth=edge_width)

    ax.text(x, y, text,
           transform=ax.transAxes,
           fontsize=fontsize,
           ha=ha, va=va,
           bbox=props)


def create_piecewise_breakpoints(thrd_1: float, thrd_2: float, shift_f: float,
                                 num_pieces: int = 6, wind_speed_max: float = 120) -> Tuple[List[float], List[float]]:
    """
    Create breakpoints for piecewise linearization matching the implementation
    in piecewise_linearize_fragility.

    Parameters
    ----------
    thrd_1 : float
        Lower threshold
    thrd_2 : float
        Upper threshold
    shift_f : float
        Base shift amount
    num_pieces : int
        Number of pieces in transition region
    wind_speed_max : float
        Maximum wind speed for the global upper bound (default: 120)

    Returns
    -------
    Tuple of (breakpoints, effective_thresholds)
    """
    # Define bounds
    global_min = 0
    global_max = wind_speed_max

    # Effective thresholds after shift
    effective_th1 = thrd_1 + shift_f
    effective_th2 = thrd_2 + shift_f

    # Create adaptive breakpoints
    breakpoints = []

    # Add global minimum (start of first flat region)
    breakpoints.append(global_min)

    # Add the first threshold (end of first flat region, start of transition)
    breakpoints.append(effective_th1)

    # Add (num_pieces - 1) points within transition region
    if num_pieces > 2:
        transition_points = np.linspace(effective_th1, effective_th2, num_pieces + 1)[1:-1]
        breakpoints.extend(transition_points.tolist())

    # Add the second threshold (end of transition, start of second flat region)
    breakpoints.append(effective_th2)

    # Add global maximum (end of second flat region)
    breakpoints.append(global_max)

    # Remove duplicates and sort
    breakpoints = sorted(list(set(breakpoints)))

    return breakpoints, (effective_th1, effective_th2)


def calculate_piecewise_fragility(breakpoints: List[float], mu: float, sigma: float,
                                  thrd_1: float, thrd_2: float, shift_f: float) -> List[float]:
    """
    Calculate failure probabilities at breakpoints for piecewise linearization.
    """
    from scipy.stats import lognorm

    fail_probs = []
    for x in breakpoints:
        z = x - shift_f  # Apply shift
        if z <= thrd_1:
            fail_probs.append(0.0)
        elif z >= thrd_2:
            fail_probs.append(1.0)
        else:
            fail_probs.append(float(lognorm.cdf(z, s=sigma, scale=np.exp(mu))))

    return fail_probs

# -----------------------------
# Main Visualization Function
# -----------------------------
def visualize_fragility_curves(
    # Parameter source options
    network_preset: Optional[str] = None,
    network_presets: Optional[List[str]] = None,  # For network_comparison plot type
    network_labels: Optional[List[str]] = None,  # Custom labels for each network
    network_colors: Optional[List[str]] = None,  # Custom colors for each network
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    thrd_1: Optional[float] = None,
    thrd_2: Optional[float] = None,
    shift_f: Optional[float] = None,

    # Hardening options
    show_hardening: bool = False,
    hardening_levels: List[float] = [10, 20, 30],
    hardening_colors: Optional[List[str]] = None,
    hardening_labels: Optional[List[str]] = None,

    # Show piecewise linearized curve
    show_piecewise: bool = False,
    num_pieces: int = 6,
    piecewise_linestyle: str = ':',
    piecewise_alpha: float = 0.8,
    piecewise_marker: str = 'o',
    piecewise_markersize: float = 4,

    # Plot type options
    plot_type: str = 'single',  # 'single', 'hardening_shift', 'comparison', 'network_comparison'

    # Curve appearance
    curve_color: str = 'blue',
    curve_linewidth: float = 2.5,
    curve_linestyle: str = '-',
    curve_alpha: float = 1.0,

    # Threshold appearance
    show_thresholds: bool = True,
    threshold_colors: List[str] = ['green', 'red'],
    threshold_linestyles: List[str] = ['--', '--'],
    threshold_linewidth: float = 1.5,
    threshold_alpha: float = 0.7,
    threshold_labels: bool = True,

    # Wind speed range
    wind_speed_min: float = 0,
    wind_speed_max: float = 120,
    wind_speed_points: int = 500,

    # Axes and labels
    custom_title: Optional[str] = None,
    xlabel: str = 'Wind gust speed (m/s)',
    ylabel: str = 'Failure probability',
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Tuple[float, float] = (-0.05, 1.05),

    # Font sizes
    title_fontsize: int = 14,
    title_fontweight: str = 'bold',
    xlabel_fontsize: int = 12,
    ylabel_fontsize: int = 12,
    tick_fontsize: int = 10,
    legend_fontsize: int = 10,
    threshold_fontsize: int = 9,

    # Legend options
    show_legend: bool = True,
    legend_loc: str = 'best',
    legend_ncol: int = 1,
    legend_frameon: bool = True,
    legend_shadow: bool = True,
    legend_fancybox: bool = True,
    legend_handlelength: float = 2.0,  # Length of legend line samples
    legend_handleheight: float = 0.7,  # Height of legend handle box
    legend_columnspacing: float = 1.0,  # Spacing between columns
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,  # Custom legend position

    # Grid options
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    grid_linestyle: str = '--',
    grid_which: str = 'major',

    # Arrow for hardening (if applicable)
    show_hardening_arrow: bool = True,
    arrow_color: str = 'black',
    arrow_style: str = 'simple',
    arrow_width: float = 2,
    arrow_fontsize: int = 11,
    arrow_text_offset: float = 0.025,  # Reduced from default 0.05

    # Information text box
    show_textbox: bool = False,
    textbox_text: Optional[str] = None,
    textbox_position: Union[str, Tuple[float, float]] = 'upper left',
    textbox_fontsize: int = 10,
    textbox_color: str = 'wheat',
    textbox_alpha: float = 0.8,

    # Figure options
    figsize: Tuple[float, float] = (10, 7),
    figure_margins: Optional[Dict[str, float]] = None,
    tight_layout: bool = True,
    tight_layout_pad: float = 1.08,

    # Save options
    save_path: Optional[str] = None,
    dpi: int = 300,
    save_bbox: str = 'tight',
    save_pad_inches: float = 0.1
):
    """
    Visualize fragility curves with flexible customization options.

    Parameters
    ----------
    network_preset : str, optional
        Network preset name to load fragility parameters from.
        Parameters are loaded from net.data.frg (mu, sigma, thrd_1, thrd_2, shift_f).
        If fragility data has multiple values (one per branch), the first value is used.
        If None, parameters must be provided manually.
    network_presets : list of str, optional
        List of network preset names for network_comparison plot type.
        Each network's fragility parameters will be loaded and plotted.
    network_labels : list of str, optional
        Custom labels for each network in network_comparison plot.
        If None, labels are auto-generated based on preset names.
    network_colors : list of str, optional
        Custom colors for each network curve in network_comparison plot.
        If None, uses default color cycle.
    mu, sigma, thrd_1, thrd_2, shift_f : float, optional
        Manual fragility curve parameters. Required if network_preset is None.
        These override values loaded from network_preset if both are provided.
    show_hardening : bool
        Whether to show multiple hardened curves
    hardening_levels : list of float
        List of hardening shift amounts (m/s)
    hardening_colors : list of str, optional
        Colors for each hardening level. If None, uses default color cycle.
    hardening_labels : list of str, optional
        Labels for each hardening level. If None, generates automatically.
    plot_type : str
        Type of plot: 'single', 'hardening_shift', 'comparison', 'network_comparison'
        - 'single': Single fragility curve
        - 'hardening_shift': Original plus hardened curves
        - 'comparison': Side-by-side original and hardened
        - 'network_comparison': Compare curves from multiple network presets
    curve_color : str
        Color of the main fragility curve
    curve_linewidth : float
        Width of the curve line
    curve_linestyle : str
        Style of the curve line ('-', '--', '-.', ':')
    curve_alpha : float
        Transparency of the curve
    show_thresholds : bool
        Whether to show threshold lines
    threshold_colors : list of str
        Colors for threshold lines [lower, upper]
    threshold_linestyles : list of str
        Styles for threshold lines
    threshold_linewidth : float
        Width of threshold lines
    threshold_alpha : float
        Transparency of threshold lines
    threshold_labels : bool
        Whether to add labels for thresholds
    wind_speed_min, wind_speed_max : float
        Range of wind speeds to plot
    wind_speed_points : int
        Number of points for wind speed array
    custom_title : str, optional
        Custom title for the plot
    xlabel, ylabel : str
        Axis labels
    xlim, ylim : tuple, optional
        Axis limits
    title_fontsize : int
        Font size for title
    title_fontweight : str
        Font weight for title
    xlabel_fontsize, ylabel_fontsize : int
        Font sizes for axis labels
    tick_fontsize : int
        Font size for tick labels
    legend_fontsize : int
        Font size for legend
    threshold_fontsize : int
        Font size for threshold labels
    show_legend : bool
        Whether to show legend
    legend_loc : str
        Legend location
    legend_ncol : int
        Number of columns in legend
    legend_frameon : bool
        Whether to show legend frame
    legend_shadow : bool
        Whether to show legend shadow
    legend_fancybox : bool
        Whether to use fancy box for legend
    legend_handlelength : float
        Length of legend line samples (default 2.0, reduce for compact legend)
    legend_handleheight : float
        Height of legend handle box (default 0.7)
    legend_columnspacing : float
        Spacing between legend columns (default 1.0)
    legend_bbox_to_anchor : tuple, optional
        Custom legend position as (x, y) in axes coordinates
    show_grid : bool
        Whether to show grid
    grid_alpha : float
        Grid transparency
    grid_linestyle : str
        Grid line style
    grid_which : str
        Which grid lines to show ('major', 'minor', 'both')
    show_hardening_arrow : bool
        Whether to show arrow indicating hardening shift
    arrow_color : str
        Color of hardening arrow
    arrow_style : str
        Style of arrow
    arrow_width : float
        Width of arrow
    arrow_fontsize : int
        Font size for arrow label
    arrow_text_offset : float
        Vertical offset for arrow text label (default 0.025, was 0.05)
    show_textbox : bool
        Whether to show information text box
    textbox_text : str, optional
        Text for text box
    textbox_position : str or tuple
        Position of text box
    textbox_fontsize : int
        Font size for text box
    textbox_color : str
        Background color of text box
    textbox_alpha : float
        Transparency of text box
    figsize : tuple
        Figure size (width, height)
    figure_margins : dict, optional
        Manual figure margins
    tight_layout : bool
        Whether to use tight layout
    tight_layout_pad : float
        Padding for tight layout
    save_path : str, optional
        Path to save figure
    dpi : int
        DPI for saved figure
    save_bbox : str
        Bounding box for saved figure
    save_pad_inches : float
        Padding for saved figure

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """

    # Load parameters from network preset if specified
    if network_preset is not None:
        if not HAS_NETWORK_FACTORY:
            raise ImportError("Network factory not available. Please provide fragility parameters manually.")

        net = make_network(network_preset)

        # Check if network has fragility data
        if hasattr(net, 'data') and hasattr(net.data, 'frg'):
            # Fragility parameters are stored as lists (one per branch)
            # Take the first value assuming all branches have the same fragility curve
            # TODO: Could be enhanced to handle different curves per branch or visualize
            # the distribution of fragility parameters across branches
            if mu is None and hasattr(net.data.frg, 'mu') and net.data.frg.mu:
                mu = net.data.frg.mu[0] if isinstance(net.data.frg.mu, list) else net.data.frg.mu
            if sigma is None and hasattr(net.data.frg, 'sigma') and net.data.frg.sigma:
                sigma = net.data.frg.sigma[0] if isinstance(net.data.frg.sigma, list) else net.data.frg.sigma
            if thrd_1 is None and hasattr(net.data.frg, 'thrd_1') and net.data.frg.thrd_1:
                thrd_1 = net.data.frg.thrd_1[0] if isinstance(net.data.frg.thrd_1, list) else net.data.frg.thrd_1
            if thrd_2 is None and hasattr(net.data.frg, 'thrd_2') and net.data.frg.thrd_2:
                thrd_2 = net.data.frg.thrd_2[0] if isinstance(net.data.frg.thrd_2, list) else net.data.frg.thrd_2
            if shift_f is None and hasattr(net.data.frg, 'shift_f') and net.data.frg.shift_f:
                shift_f = net.data.frg.shift_f[0] if isinstance(net.data.frg.shift_f, list) else net.data.frg.shift_f
        else:
            # Try to load default parameters from WindConfig if no fragility data in network
            try:
                from core.wind_config import WindConfig as WindConfigClass
                wind_config = WindConfigClass()
                if mu is None:
                    mu = wind_config.data.frg.mu
                if sigma is None:
                    sigma = wind_config.data.frg.sigma
                if thrd_1 is None:
                    thrd_1 = wind_config.data.frg.thrd_1
                if thrd_2 is None:
                    thrd_2 = wind_config.data.frg.thrd_2
                if shift_f is None:
                    shift_f = wind_config.data.frg.shift_f
            except ImportError:
                pass  # Will be caught by validation below

    # Validate that all parameters are provided (except for network_comparison plot type)
    if plot_type != 'network_comparison' and any(param is None for param in [mu, sigma, thrd_1, thrd_2, shift_f]):
        raise ValueError("All fragility parameters...")

    # Generate wind speed range
    wind_speeds = np.linspace(wind_speed_min, wind_speed_max, wind_speed_points)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot based on type
    if plot_type == 'single' or (plot_type == 'hardening_shift' and not show_hardening):
        # Single curve
        pof_values = calculate_fragility_pof(wind_speeds, mu, sigma, thrd_1, thrd_2, shift_f)

        plot_single_fragility_curve(
            ax, wind_speeds, pof_values,
            thrd_1, thrd_2, shift_f,
            curve_color=curve_color,
            curve_linewidth=curve_linewidth,
            curve_label='Fragility Curve',
            show_thresholds=show_thresholds,
            threshold_colors=threshold_colors,
            threshold_linestyles=threshold_linestyles,
            threshold_linewidth=threshold_linewidth,
            threshold_alpha=threshold_alpha,
            threshold_labels=threshold_labels,
            threshold_fontsize=threshold_fontsize
        )

        if custom_title is None:
            custom_title = 'Fragility Curve'

    elif plot_type == 'hardening_shift' and show_hardening:
        # Multiple hardened curves
        # Set default colors if not provided
        if hardening_colors is None:
            default_colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray']
            hardening_colors = [curve_color] + default_colors[:len(hardening_levels)]

        # Set default labels if not provided
        if hardening_labels is None:
            # Use compact labels if legend space is constrained
            if legend_handlelength < 2.0 or legend_loc in ['upper right', 'right', 'center right']:
                # Shorter labels for space-constrained legends
                hardening_labels = ['Original'] + [f'Hardened (+{h:.0f} m/s)' for h in hardening_levels]
            else:
                # Full descriptive labels
                hardening_labels = ['Original fragility curve'] + [f'Fragility curve after hardening ({h:.0f} m/s shift)' for h in hardening_levels]

        # Plot original curve
        pof_original = calculate_fragility_pof(wind_speeds, mu, sigma, thrd_1, thrd_2, shift_f)
        plot_single_fragility_curve(
            ax, wind_speeds, pof_original,
            thrd_1, thrd_2, shift_f,
            curve_color=hardening_colors[0],
            curve_linewidth=curve_linewidth,
            curve_label=hardening_labels[0],
            show_thresholds=False  # Will add thresholds separately
        )

        # Plot hardened curves
        for i, hardening in enumerate(hardening_levels, 1):
            pof_hardened = calculate_fragility_pof(
                wind_speeds, mu, sigma, thrd_1, thrd_2, shift_f, hardening
            )

            color = hardening_colors[i] if i < len(hardening_colors) else 'gray'
            label = hardening_labels[i] if i < len(hardening_labels) else f'{hardening:.0f} m/s hardening'

            ax.plot(wind_speeds, pof_hardened,
                   color=color,
                   linewidth=curve_linewidth,
                   linestyle=curve_linestyle,
                   alpha=curve_alpha,
                   label=label)

        # Add threshold lines for original curve
        if show_thresholds:
            ax.axvline(thrd_1 + shift_f,
                      color=threshold_colors[0],
                      linestyle=threshold_linestyles[0],
                      linewidth=threshold_linewidth,
                      alpha=threshold_alpha,
                      label=f'Original Threshold 1 ({thrd_1 + shift_f:.1f} m/s)')

            ax.axvline(thrd_2 + shift_f,
                      color=threshold_colors[1],
                      linestyle=threshold_linestyles[1],
                      linewidth=threshold_linewidth,
                      alpha=threshold_alpha,
                      label=f'Original Threshold 2 ({thrd_2 + shift_f:.1f} m/s)')

        # Add arrow showing hardening direction
        if show_hardening_arrow and len(hardening_levels) > 0:
            # Find wind speed at 50% PoF for arrow placement
            target_pof = 0.5
            idx_original = np.argmin(np.abs(pof_original - target_pof))
            wind_original = wind_speeds[idx_original]
            wind_hardened = wind_original + hardening_levels[0]

            add_hardening_arrow(
                ax, wind_original, wind_hardened, target_pof,
                arrow_color=arrow_color,
                arrow_style=arrow_style,
                arrow_width=arrow_width,
                text_label='Hardening\nShift',
                text_fontsize=arrow_fontsize,
                text_offset=arrow_text_offset
            )

        if custom_title is None:
            custom_title = 'Fragility Curve Shift due to Line Hardening'

    elif plot_type == 'comparison':
        # Side-by-side comparison (would need subplot adjustment)
        # For now, just plot original and one hardened curve
        if len(hardening_levels) == 0:
            hardening_levels = [20]  # Default hardening

        pof_original = calculate_fragility_pof(wind_speeds, mu, sigma, thrd_1, thrd_2, shift_f)
        pof_hardened = calculate_fragility_pof(
            wind_speeds, mu, sigma, thrd_1, thrd_2, shift_f, hardening_levels[0]
        )

        ax.plot(wind_speeds, pof_original,
               color=curve_color,
               linewidth=curve_linewidth,
               linestyle='-',
               label='Original fragility curve')

        ax.plot(wind_speeds, pof_hardened,
               color='red' if curve_color != 'red' else 'green',
               linewidth=curve_linewidth,
               linestyle='--',
               label=f'Fragility curve after hardening ({hardening_levels[0]:.0f} m/s shift)')

        if show_thresholds:
            # Original thresholds
            ax.axvline(thrd_1 + shift_f,
                      color=threshold_colors[0],
                      linestyle=':',
                      linewidth=threshold_linewidth * 0.8,
                      alpha=threshold_alpha * 0.7)

            ax.axvline(thrd_2 + shift_f,
                      color=threshold_colors[1],
                      linestyle=':',
                      linewidth=threshold_linewidth * 0.8,
                      alpha=threshold_alpha * 0.7)

            # Hardened thresholds
            ax.axvline(thrd_1 + shift_f + hardening_levels[0],
                      color=threshold_colors[0],
                      linestyle=threshold_linestyles[0],
                      linewidth=threshold_linewidth,
                      alpha=threshold_alpha)

            ax.axvline(thrd_2 + shift_f + hardening_levels[0],
                      color=threshold_colors[1],
                      linestyle=threshold_linestyles[1],
                      linewidth=threshold_linewidth,
                      alpha=threshold_alpha)

        if custom_title is None:
            custom_title = 'Original vs Hardened Fragility Curves'

    elif plot_type == 'network_comparison':
        # Compare fragility curves from multiple network presets
        if network_presets is None or len(network_presets) == 0:
            raise ValueError("network_presets must be provided for network_comparison plot type")

        # Set default colors if not provided
        if network_colors is None:
            default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            network_colors = default_colors[:len(network_presets)]

        # Set default labels if not provided
        if network_labels is None:
            network_labels = []
            for preset in network_presets:
                if 'transmission' in preset.lower() or 'tn' in preset.lower() or 'GB_Transmission' in preset:
                    network_labels.append('TN OHL')
                elif 'distribution' in preset.lower() or 'dn' in preset.lower() or 'Manchester' in preset:
                    network_labels.append('DN OHL')
                else:
                    network_labels.append(preset.replace('_', ' ').split()[0])

        # Store first network's thresholds for plotting later
        first_net_thrd_1, first_net_thrd_2, first_net_shift_f = None, None, None
        first_net_color = None

        # Plot each network's fragility curve
        for i, (preset, label, color) in enumerate(zip(network_presets, network_labels, network_colors)):
            # Load parameters for this network
            if not HAS_NETWORK_FACTORY:
                raise ImportError("Network factory required for network comparison")

            net = make_network(preset)

            # Get fragility parameters from this network
            if hasattr(net, 'data') and hasattr(net.data, 'frg'):
                net_mu = net.data.frg.mu[0] if isinstance(net.data.frg.mu, list) else net.data.frg.mu
                net_sigma = net.data.frg.sigma[0] if isinstance(net.data.frg.sigma, list) else net.data.frg.sigma
                net_thrd_1 = net.data.frg.thrd_1[0] if isinstance(net.data.frg.thrd_1,
                                                                  list) else net.data.frg.thrd_1
                net_thrd_2 = net.data.frg.thrd_2[0] if isinstance(net.data.frg.thrd_2,
                                                                  list) else net.data.frg.thrd_2
                net_shift_f = net.data.frg.shift_f[0] if isinstance(net.data.frg.shift_f,
                                                                    list) else net.data.frg.shift_f
            else:
                raise ValueError(f"Network preset '{preset}' doesn't have fragility data")

            # Store first network's threshold values for later plotting
            if i == 0:
                first_net_thrd_1 = net_thrd_1
                first_net_thrd_2 = net_thrd_2
                first_net_shift_f = net_shift_f
                first_net_color = color

            # Calculate smooth fragility curve
            pof_values = calculate_fragility_pof(wind_speeds, net_mu, net_sigma,
                                                 net_thrd_1, net_thrd_2, net_shift_f)

            # Plot smooth curve
            ax.plot(wind_speeds, pof_values,
                    color=color,
                    linewidth=curve_linewidth,
                    linestyle='-',
                    label=label,
                    alpha=1.0)

            # Add piecewise linearization if requested
            if show_piecewise:
                # Create breakpoints
                breakpoints, effective_thresholds = create_piecewise_breakpoints(
                    net_thrd_1, net_thrd_2, net_shift_f, num_pieces, wind_speed_max
                )

                # Calculate probabilities at breakpoints
                piecewise_probs = calculate_piecewise_fragility(
                    breakpoints, net_mu, net_sigma, net_thrd_1, net_thrd_2, net_shift_f
                )

                # Plot piecewise linear approximation
                ax.plot(breakpoints, piecewise_probs,
                        color=color,
                        linewidth=curve_linewidth * 0.8,
                        linestyle=piecewise_linestyle,
                        marker=piecewise_marker,
                        markersize=piecewise_markersize,
                        label=f'{label} ({num_pieces + 2} pieces)',
                        alpha=piecewise_alpha)

                # REMOVED: No threshold plotting inside the loop

        # Add threshold lines AFTER all curves are plotted (so they appear at the end of legend)
        if show_thresholds and first_net_thrd_1 is not None:
            # Use the first network's thresholds
            ax.axvline(first_net_thrd_1 + first_net_shift_f,
                       color=first_net_color,
                       linestyle='--',
                       linewidth=threshold_linewidth * 0.5,
                       alpha=threshold_alpha * 0.5,
                       label='Lower threshold')

            ax.axvline(first_net_thrd_2 + first_net_shift_f,
                       color=first_net_color,
                       linestyle='--',
                       linewidth=threshold_linewidth * 0.5,
                       alpha=threshold_alpha * 0.5,
                       label='Upper threshold')

        if custom_title is None:
            if show_piecewise:
                custom_title = f'Fragility Curves with Piecewise Linearization ({num_pieces} pieces)'
            else:
                custom_title = 'Fragility Curve Comparison'

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    if custom_title:
        ax.set_title(custom_title, fontsize=title_fontsize, fontweight=title_fontweight)

    # Set axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        # Auto-adjust x-axis based on content
        if show_hardening and len(hardening_levels) > 0:
            max_x = max(wind_speed_max, thrd_2 + shift_f + max(hardening_levels) + 10)
            ax.set_xlim(wind_speed_min, max_x)

    ax.set_ylim(ylim)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Grid
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle, which=grid_which)

    # Legend
    if show_legend:
        legend_kwargs = {
            'loc': legend_loc,
            'fontsize': legend_fontsize,
            'ncol': legend_ncol,
            'frameon': legend_frameon,
            'shadow': legend_shadow,
            'fancybox': legend_fancybox,
            'handlelength': legend_handlelength,
            'handleheight': legend_handleheight,
            'columnspacing': legend_columnspacing
        }

        # Add custom position if specified
        if legend_bbox_to_anchor is not None:
            legend_kwargs['bbox_to_anchor'] = legend_bbox_to_anchor

        ax.legend(**legend_kwargs)

    # Add text box if requested
    if show_textbox:
        if textbox_text is None:
            if plot_type == 'hardening_shift':
                textbox_text = ('Line hardening shifts the fragility\n'
                              'curve rightward, requiring higher\n'
                              'wind speeds to cause failure')
            else:
                textbox_text = ('Fragility curve models the probability\n'
                              'of line failure under wind loading')

        add_info_textbox(ax, textbox_text,
                        position=textbox_position,
                        fontsize=textbox_fontsize,
                        box_color=textbox_color,
                        box_alpha=textbox_alpha)

    # Apply manual figure margins if specified
    if figure_margins:
        plt.subplots_adjust(
            left=figure_margins.get('left', 0.1),
            right=figure_margins.get('right', 0.9),
            top=figure_margins.get('top', 0.9),
            bottom=figure_margins.get('bottom', 0.1)
        )
    elif tight_layout:
        plt.tight_layout(pad=tight_layout_pad)

    # Save if requested
    if save_path:
        plt.savefig(save_path,
                   dpi=dpi,
                   bbox_inches=save_bbox,
                   pad_inches=save_pad_inches)
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig, ax


# -----------------------------
# Additional Utility Functions
# -----------------------------
def compare_fragility_parameters(
    parameter_sets: List[Dict[str, float]],
    labels: List[str],
    figsize: Tuple[float, float] = (15, 5),
    wind_speed_range: Tuple[float, float] = (0, 120),
    **kwargs
):
    """
    Compare multiple fragility curves with different parameters.

    Parameters
    ----------
    parameter_sets : list of dict
        List of dictionaries containing fragility parameters
        Each dict should have keys: 'mu', 'sigma', 'thrd_1', 'thrd_2', 'shift_f'
    labels : list of str
        Labels for each parameter set
    figsize : tuple
        Figure size
    wind_speed_range : tuple
        Range of wind speeds to plot
    **kwargs
        Additional keyword arguments passed to visualization

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    n_sets = len(parameter_sets)
    fig, axes = plt.subplots(1, n_sets, figsize=figsize, sharey=True)

    if n_sets == 1:
        axes = [axes]

    wind_speeds = np.linspace(wind_speed_range[0], wind_speed_range[1], 500)

    for i, (params, label, ax) in enumerate(zip(parameter_sets, labels, axes)):
        # Calculate PoF
        pof = calculate_fragility_pof(
            wind_speeds,
            params['mu'],
            params['sigma'],
            params['thrd_1'],
            params['thrd_2'],
            params['shift_f']
        )

        # Plot
        plot_single_fragility_curve(
            ax, wind_speeds, pof,
            params['thrd_1'],
            params['thrd_2'],
            params['shift_f'],
            curve_color=kwargs.get('curve_color', 'blue'),
            curve_linewidth=kwargs.get('curve_linewidth', 2.5),
            show_thresholds=kwargs.get('show_thresholds', True)
        )

        ax.set_xlabel('Wind Speed (m/s)', fontsize=kwargs.get('xlabel_fontsize', 12))
        if i == 0:
            ax.set_ylabel('Failure Probability', fontsize=kwargs.get('ylabel_fontsize', 12))

        ax.set_title(label, fontsize=kwargs.get('title_fontsize', 14))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=kwargs.get('legend_fontsize', 10))

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
    Example usage of fragility curve visualization.
    """

    # Plot both original and hardened fragility curve (from a single network preset):
    # visualize_fragility_curves(
    #     plot_type='hardening_shift',
    #
    #     # select network preset:
    #     network_preset="Manchester_distribution_network_Kearsley",
    #     custom_title="Fragility Curve of DN OHL",
    #     # network_preset="GB_Transmission_Network_29_Bus",
    #     # custom_title="Fragility Curve of TN OHL",
    #
    #     show_hardening=True,
    #     hardening_levels=[15],
    #     hardening_colors=['#1f77b4', '#2ca02c'],
    #     show_hardening_arrow=True,
    #     arrow_text_offset=0.025,
    #     arrow_fontsize=13,
    #     curve_linewidth=3,
    #     show_thresholds=False,
    #     threshold_linewidth=2,
    #     threshold_linestyles=[':', ':'],
    #     threshold_colors=['darkblue', 'darkblue'],
    #     # show_textbox=False,
    #     # textbox_position='upper right',
    #     # textbox_color='lightblue',
    #     title_fontsize=16,
    #     xlabel_fontsize=14,
    #     ylabel_fontsize=14,
    #     legend_fontsize=12,
    #     legend_loc='upper left',
    #     legend_handlelength=1.5,
    #     figsize=(8, 8),
    #     grid_alpha=0.2,
    #     wind_speed_max=70,
    #     # save_path=None,
    #     dpi=300
    # )

    # Plot fragility curves from multiple network presets (without showing hardened curve)

    visualize_fragility_curves(
        plot_type='network_comparison',
        network_presets=[
            "GB_Transmission_Network_29_Bus",
            "Manchester_distribution_network_Kearsley",
        ],

        network_colors=['green', 'orange'],
        show_thresholds=False,
        threshold_linewidth=3,

        custom_title='Fragility Curves Visualisation',
        curve_linewidth=2,

        # Enable piecewise visualization
        show_piecewise=True,
        num_pieces=5,  # Number of pieces between thrd_1 and thrd_2
        piecewise_linestyle=':',
        piecewise_alpha=0.8,
        piecewise_marker='o',
        piecewise_markersize=5,

        title_fontsize=17,
        xlabel_fontsize=16,
        ylabel_fontsize=16,
        tick_fontsize=14,
        legend_fontsize=15,
        legend_loc='upper left',
        legend_handlelength=1.5,
        figsize=(8, 4.5),
        wind_speed_max=80
    )


    # ==========================================================================

    # Example 1: Basic fragility curve from network preset
    # Note: Fragility parameters are loaded from net.data.frg
    # print("Example 1: Basic fragility curve from network preset")
    # visualize_fragility_curves(
    #     network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
    #     custom_title="Overhead Line Fragility Curve",
    #     show_textbox=True,
    #     textbox_text="Log-normal fragility function\nfor overhead transmission lines"
    # )

    # Example 2: Fragility curve with manual parameters
    # print("\nExample 2: Manual parameters")
    # visualize_fragility_curves(
    #     mu=3.5,
    #     sigma=0.3,
    #     thrd_1=20,
    #     thrd_2=80,
    #     shift_f=10,
    #     custom_title="Custom Fragility Curve",
    #     curve_color='red',
    #     threshold_colors=['blue', 'orange']
    # )

    # Example 3: Hardening shift visualization
    # print("\nExample 3: Hardening shift visualization")
    # visualize_fragility_curves(
    #     network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
    #     plot_type='hardening_shift',
    #     show_hardening=True,
    #     hardening_levels=[10, 20, 30],
    #     hardening_colors=['blue', 'green', 'orange', 'red'],
    #     show_hardening_arrow=True,
    #     show_textbox=True,
    #     custom_title="Impact of Line Hardening on Fragility",
    #     figsize=(12, 8)
    # )

    # Example 4: Comparison plot
    # print("\nExample 4: Comparison of original and hardened curves")
    # visualize_fragility_curves(
    #     network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
    #     plot_type='comparison',
    #     hardening_levels=[25],
    #     custom_title="Fragility Curve: Original vs Hardened (25 m/s)",
    #     show_thresholds=True
    # )

    # Example 5: Customized appearance
    # print("\nExample 5: Highly customized visualization")
    # visualize_fragility_curves(
    #     network_preset="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
    #     plot_type='hardening_shift',
    #     show_hardening=True,
    #     hardening_levels=[15, 30, 45],
    #     hardening_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    #     curve_linewidth=3,
    #     threshold_linewidth=2,
    #     threshold_linestyles=[':', ':'],
    #     show_textbox=True,
    #     textbox_position='upper right',
    #     textbox_color='lightblue',
    #     title_fontsize=16,
    #     xlabel_fontsize=14,
    #     ylabel_fontsize=14,
    #     legend_fontsize=12,
    #     legend_loc='center right',
    #     figsize=(14, 9),
    #     grid_alpha=0.2,
    #     wind_speed_max=140,
    #     save_path="fragility_curves_customized.png",
    #     dpi=300
    # )

    # Example 6: Parameter comparison
    # print("\nExample 6: Comparing different fragility parameters")
    # parameter_sets = [
    #     {'mu': 3.5, 'sigma': 0.3, 'thrd_1': 20, 'thrd_2': 80, 'shift_f': 10},
    #     {'mu': 3.8, 'sigma': 0.25, 'thrd_1': 25, 'thrd_2': 85, 'shift_f': 12},
    #     {'mu': 3.2, 'sigma': 0.35, 'thrd_1': 18, 'thrd_2': 75, 'shift_f': 8}
    # ]
    # labels = ['Standard OHL', 'Reinforced OHL', 'Vulnerable OHL']
    #
    # compare_fragility_parameters(
    #     parameter_sets,
    #     labels,
    #     figsize=(15, 5),
    #     show_thresholds=True,
    #     curve_color='navy'
    # )

    # Example 7: Network comparison - TN vs DN fragility curves
    # print("\nExample 7: Comparing TN and DN network fragility curves")
    # visualize_fragility_curves(
    #     plot_type='network_comparison',
    #     network_presets=[
    #         "GB_Transmission_Network_29_Bus",
    #         "Manchester_distribution_network_Kearsley"
    #     ],
    #     # Auto-generates labels as ['TN OHL', 'DN OHL']
    #     show_thresholds=False,  # Clean comparison
    #     custom_title='Fragility Curves: TN vs DN OHL',
    #     curve_linewidth=3,
    #     legend_loc='upper left',
    #     legend_handlelength=1.5,
    #     figsize=(10, 7),
    #     wind_speed_max=70
    # )