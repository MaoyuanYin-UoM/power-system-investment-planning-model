"""
Visualization script for wind gust speed distribution (Weibull).

This script visualizes the Weibull distribution used to model wind gust speeds
in windstorm events. The distribution parameters (shape k and scale λ) are
loaded from a specified windstorm preset configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.windstorm import WindConfig
from factories.windstorm_factory import make_windstorm


def load_windstorm_config(preset_name: str):
    """
    Load windstorm configuration from a preset.

    Parameters
    ----------
    preset_name : str
        Name of the windstorm preset to load

    Returns
    -------
    WindstormConfig
        Loaded windstorm configuration object
    """
    ws = make_windstorm(preset_name)
    return ws


def get_weibull_parameters(ws) -> tuple:
    """
    Extract Weibull distribution parameters from windstorm configuration.

    Parameters
    ----------
    ws : an instance of WindConfig

    Returns
    -------
    tuple
        (shape, scale) parameters of the Weibull distribution
    """
    shape = ws.data.WS.event.gust_weibull_shape  # k
    scale = ws.data.WS.event.gust_weibull_scale  # λ (m/s)
    return shape, scale


def plot_weibull_distribution(
        shape,  # Can be a single float or a list of floats
        scale,  # Can be a single float or a list of floats
        labels=None,  # Optional labels for each distribution
        title: str = "Wind Gust Speed Distribution (Weibull)",
        xlabel: str = "Wind Gust Speed (m/s)",
        ylabel: str = "Probability Density",
        figsize: tuple = (10, 6),
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        tick_fontsize: int = 10,
        legend_fontsize: int = 10,
        grid: bool = True,
        show_cdf: bool = True,
        show_stats: bool = True,
        output_path: str = None
):
    """
    Plot the Weibull distribution(s) for wind gust speeds.

    Parameters
    ----------
    shape : float or list of float
        Shape parameter(s) (k) of the Weibull distribution(s)
    scale : float or list of float
        Scale parameter(s) (λ) of the Weibull distribution(s) (m/s)
    labels : list of str, optional
        Labels for each distribution. If None, default labels will be generated.
    title : str, optional
        Title of the plot
    xlabel : str, optional
        Label for x-axis
    ylabel : str, optional
        Label for y-axis (PDF)
    figsize : tuple, optional
        Figure size (width, height)
    title_fontsize : int, optional
        Font size for title
    label_fontsize : int, optional
        Font size for axis labels
    tick_fontsize : int, optional
        Font size for tick labels
    legend_fontsize : int, optional
        Font size for legend
    grid : bool, optional
        Whether to show grid
    show_cdf : bool, optional
        Whether to show CDF on a secondary y-axis
    show_stats : bool, optional
        Whether to show statistics text box (only for single distribution)
    output_path : str, optional
        Path to save the figure. If None, the figure is displayed.
    """
    # Convert single values to lists for uniform handling
    if not isinstance(shape, list):
        shape = [shape]
    if not isinstance(scale, list):
        scale = [scale]

    # Validate input
    if len(shape) != len(scale):
        raise ValueError(f"shape and scale must have the same length. Got {len(shape)} and {len(scale)}.")

    num_distributions = len(shape)

    # Generate default labels if not provided
    if labels is None:
        if num_distributions == 1:
            labels = [f'k={shape[0]}, λ={scale[0]} m/s']
        else:
            labels = [f'k={shape[i]}, λ={scale[i]} m/s' for i in range(num_distributions)]
    elif len(labels) != num_distributions:
        raise ValueError(f"labels must have the same length as shape and scale. Got {len(labels)}.")

    # Define colors for multiple distributions
    colors_pdf = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
                  'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors_cdf = ['tab:orange', 'tab:cyan', 'tab:pink', 'tab:brown', 'tab:gray',
                  'tab:olive', 'tab:red', 'tab:purple', 'tab:green']

    # Determine x-axis range (use the maximum scale for range)
    x_max = max(scale) * 3
    x = np.linspace(0, x_max, 1000)

    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)

    # Storage for legend
    lines_pdf = []
    lines_cdf = []

    # Plot each distribution
    for i in range(num_distributions):
        # Create Weibull distribution object
        weibull_dist = weibull_min(c=shape[i], scale=scale[i])

        # Calculate PDF
        pdf = weibull_dist.pdf(x)

        # Plot PDF
        color_pdf = colors_pdf[i % len(colors_pdf)]
        line_pdf = ax1.plot(x, pdf, color=color_pdf, linewidth=2,
                            label=f'{labels[i]}')
        lines_pdf.extend(line_pdf)

    # Configure primary y-axis (PDF)
    ax1.set_xlabel(xlabel, fontsize=label_fontsize)
    ax1.set_ylabel(ylabel, fontsize=label_fontsize)
    ax1.tick_params(axis='both', labelsize=tick_fontsize)

    # Plot CDF on secondary y-axis if requested
    if show_cdf:
        ax2 = ax1.twinx()

        for i in range(num_distributions):
            # Create Weibull distribution object
            weibull_dist = weibull_min(c=shape[i], scale=scale[i])

            # Calculate CDF
            cdf = weibull_dist.cdf(x)

            # Plot CDF
            color_cdf = colors_cdf[i % len(colors_cdf)]
            line_cdf = ax2.plot(x, cdf, color=color_cdf, linewidth=2,
                                linestyle='--', label=f'CDF: {labels[i]}')
            lines_cdf.extend(line_cdf)

        ax2.set_ylabel('Cumulative Probability', fontsize=label_fontsize)
        ax2.tick_params(axis='y', labelsize=tick_fontsize)
        ax2.set_ylim([0, 1.05])

        # Combine legends
        all_lines = lines_pdf + lines_cdf
        all_labels = [line.get_label() for line in all_lines]
        ax1.legend(all_lines, all_labels, loc='best', fontsize=legend_fontsize)
    else:
        ax1.legend(loc='best', fontsize=legend_fontsize)

    # Add grid
    if grid:
        ax1.grid(True, alpha=0.3)

    # Set title
    ax1.set_title(title, fontsize=title_fontsize, fontweight='bold')

    # Add statistical information text box (only for single distribution)
    if show_stats and num_distributions == 1:
        weibull_dist = weibull_min(c=shape[0], scale=scale[0])
        mean_speed = weibull_dist.mean()
        median_speed = weibull_dist.median()
        std_speed = weibull_dist.std()

        textstr = f'Mean: {mean_speed:.2f} m/s\nMedian: {median_speed:.2f} m/s\nStd: {std_speed:.2f} m/s'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=tick_fontsize,
                 verticalalignment='top', bbox=props)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# Script Entry
# ============================================================================

if __name__ == "__main__":
    # Configuration
    preset_name = "windstorm_29_bus_GB_transmission_network"

    # Load windstorm configuration
    print(f"Loading windstorm preset: {preset_name}")
    ws = load_windstorm_config(preset_name)

    # Get Weibull parameters
    shape, scale = get_weibull_parameters(ws)
    print(f"Weibull parameters - Shape (k): {shape}, Scale (λ): {scale} m/s")

    # Check gust model
    gust_model = ws.data.WS.event.gust_model
    if gust_model != 'constant_weibull':
        print(f"Warning: gust_model is '{gust_model}', not 'constant_weibull'")
        print("The visualization assumes Weibull distribution.")

    # Create visualization
    print("Creating visualization...")
    plot_weibull_distribution(
        # shape=shape,
        shape=[2.0, 2.0],
        # scale=scale,
        scale=[30, 35],
        title="Wind Gust Speed Distribution (Weibull)",
        xlabel="Wind Gust Speed (m/s)",
        ylabel="Probability Density",
        figsize=(10, 6),
        title_fontsize=16,
        label_fontsize=14,
        tick_fontsize=13,
        legend_fontsize=14,
        grid=True,
        show_cdf=False,
        output_path=None,  # The figure will not be saved if set 'None'
    )

    print("Done!")