def plot_pareto_front_from_excel(
        excel_path="Post-processed_Data_for_Plots/investment_costs_vs_resilience_metrics_at_20_hrdn_shift.xlsx",
        plot_types=None,
        custom_titles=None,
        overlay_plots=False,
        overlay_line_styles=None,
        show_threshold_labels=True,
        annotation_spacing=1,
        figure_size=(12, 8),
        title_fontsize=14,
        label_fontsize=12,
        tick_fontsize=10,
        annotation_fontsize=9,
        legend_fontsize=11,
        save_path=None,
        marker_size=100,
        line_style='-',
        line_width=2,
        marker_style='o',
        color_scheme='viridis',
        show_grid=True,
        grid_alpha=0.3,
        dpi=300
):
    """
    Plot Pareto front from Excel data showing trade-off between investment costs and resilience metrics.

    Parameters
    ----------
    excel_path : str
        Path to the Excel file containing the data (relative to project root).
    plot_types : str, list of str, or None
        Type(s) of plot to generate. Options:
        - 'total_investment': Total investment cost vs resilience metric
        - 'line_hardening': Line hardening cost vs resilience metric
        - 'dg_installation': DG installation cost vs resilience metric
        - 'ess_installation': ESS installation cost vs resilience metric
        - List of above options: e.g., ['total_investment', 'line_hardening']
        - None: defaults to 'total_investment'
    custom_titles : str, dict, or None
        Custom titles for the plots:
        - str: Used as title for single plot or overlay plot
        - dict: Maps plot_type to title, e.g., {'total_investment': 'My Custom Title'}
        - None: Uses default titles
    overlay_plots : bool
        If True and multiple plot_types specified, overlay all plots on single axes.
        If False, create separate subplots for each plot type.
    overlay_line_styles : str, list, or None
        Line styles for overlaid plots (only used when overlay_plots=True):
        - str: Same line style for all plots (e.g., '-' for all solid)
        - list: List of line styles matching plot_types order
        - None: Auto-assign different styles for distinction
    show_threshold_labels : bool
        Whether to show resilience_metric_threshold values as labels near data points.
    annotation_spacing : int
        Show labels every nth data point to avoid overcrowding (1 = all points).
    figure_size : tuple
        Figure size as (width, height).
    title_fontsize : int
        Font size for plot title.
    label_fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for axis tick labels.
    annotation_fontsize : int
        Font size for threshold value annotations.
    legend_fontsize : int
        Font size for legend.
    save_path : str or None
        Path to save the figure. If None, figure is not saved.
    marker_size : int
        Size of data point markers.
    line_style : str
        Style of the line for non-overlay plots ('-', '--', '-.', ':').
    line_width : float
        Width of the line.
    marker_style : str
        Style of markers ('o', 's', '^', 'D', etc.).
    color_scheme : str or list
        Colormap name or list of colors for different plots.
    show_grid : bool
        Whether to show grid.
    grid_alpha : float
        Grid transparency (0-1).
    dpi : int
        DPI for saved figure.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes or list
        Figure and axes objects. If multiple plots and not overlaid, ax is a list of axes.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import os

    # Get project root and construct full path
    # Handle the case where this function might be in a subfolder
    current_file = Path(__file__).absolute()

    # Try to find the project root by looking for key directories/files
    # Start from current location and move up
    potential_root = current_file.parent
    max_levels_up = 3  # Maximum levels to search upward

    for _ in range(max_levels_up):
        # Check if this looks like the project root
        # (has Input_Data or Post-processed_Data_for_Plots folders)
        if (potential_root / "Post-processed_Data_for_Plots").exists() or \
                (potential_root / "Input_Data").exists() or \
                (potential_root / "main.py").exists():
            project_root = potential_root
            break
        potential_root = potential_root.parent
    else:
        # If not found, assume one level up from current file location
        project_root = current_file.parent.parent if current_file.parent.name in ['visualizations',
                                                                                  'visualization'] else current_file.parent

    full_path = project_root / excel_path

    if not full_path.exists():
        # Provide helpful error message with debugging info
        error_msg = f"Excel file not found at: {full_path}\n"
        error_msg += f"Current script location: {current_file}\n"
        error_msg += f"Detected project root: {project_root}\n"
        error_msg += f"Looking for: {excel_path}\n"

        # Check if the Post-processed_Data_for_Plots directory exists
        post_proc_dir = project_root / "Post-processed_Data_for_Plots"
        if post_proc_dir.exists():
            error_msg += f"\nPost-processed_Data_for_Plots directory found at: {post_proc_dir}\n"
            error_msg += f"Files in directory: {list(post_proc_dir.glob('*.xlsx'))}"
        else:
            error_msg += f"\nPost-processed_Data_for_Plots directory NOT found at expected location"

        raise FileNotFoundError(error_msg)

    # Read Excel data (skip first row, use second row as header)
    df = pd.read_excel(full_path, skiprows=2)

    # Clean column names (remove any leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Handle plot_types parameter
    if plot_types is None:
        plot_types = ['total_investment']  # Default to total investment
    elif isinstance(plot_types, str):
        plot_types = [plot_types]  # Convert single string to list

    # Validate plot types
    valid_types = ['total_investment', 'line_hardening', 'dg_installation', 'ess_installation']
    for ptype in plot_types:
        if ptype not in valid_types:
            raise ValueError(f"Invalid plot type: {ptype}. Must be one of {valid_types}")

    # Define default plot configurations
    plot_configs = {
        'total_investment': {
            'y_col': 'total_investment_cost',
            'default_title': 'Pareto Front: Total Investment Cost vs Resilience Metric Trade-off',
            'ylabel': 'Total Investment Cost (£ Million)',
            'color': '#2E7D32'  # Dark green
        },
        'line_hardening': {
            'y_col': 'line_hardening_cost',
            'default_title': 'Pareto Front: Line Hardening Cost vs Resilience Metric Trade-off',
            'ylabel': 'Line Hardening Cost (£ Million)',
            'color': '#1976D2'  # Dark blue
        },
        'dg_installation': {
            'y_col': 'dg_installation_cost',
            'default_title': 'Pareto Front: DG Installation Cost vs Resilience Metric Trade-off',
            'ylabel': 'DG Installation Cost (£ Million)',
            'color': '#F57C00'  # Dark orange
        },
        'ess_installation': {
            'y_col': 'ess_installation_cost',
            'default_title': 'Pareto Front: ESS Installation Cost vs Resilience Metric Trade-off',
            'ylabel': 'ESS Installation Cost (£ Million)',
            'color': '#7B1FA2'  # Dark purple
        }
    }

    # Handle custom_titles parameter
    if custom_titles is None:
        custom_titles = {}
    elif isinstance(custom_titles, str):
        # If single string provided
        if len(plot_types) == 1:
            # Single plot, single title
            custom_titles = {plot_types[0]: custom_titles}
        elif overlay_plots:
            # Multiple plots overlaid - single title makes sense
            # Keep as string for overlay plot title
            pass
        else:
            # Multiple separate plots but only one title provided
            raise ValueError("Single custom title string provided but multiple separate plots requested. "
                             "Please provide a dictionary mapping plot types to titles, or set overlay_plots=True.")

    # Apply custom titles to configurations (only for non-overlay or dict titles)
    if isinstance(custom_titles, dict):
        for ptype in plot_types:
            if ptype in custom_titles:
                plot_configs[ptype]['title'] = custom_titles[ptype]
            else:
                plot_configs[ptype]['title'] = plot_configs[ptype]['default_title']
    else:
        # For string titles with overlay or single plot
        for ptype in plot_types:
            plot_configs[ptype]['title'] = plot_configs[ptype]['default_title']

    # X-axis column (resilience metric)
    x_col = 'resilience_metric_(total_dn_eens_ws_scn)'
    threshold_col = 'resilience_metric_threshold'

    # Convert costs from £ to £ Million for better readability
    for config in plot_configs.values():
        y_col = config['y_col']
        if y_col in df.columns:
            df[f"{y_col}_millions"] = df[y_col] / 1e6

    # Convert resilience metric from MWh to GWh for better readability
    df['eens_gwh'] = df[x_col] / 1000

    # Sort by resilience metric for proper line connection
    df = df.sort_values('eens_gwh')

    # Create figure based on overlay option and number of plots
    n_plots = len(plot_types)

    if overlay_plots and n_plots > 1:
        # Create single axes for overlay
        fig, ax = plt.subplots(figsize=figure_size)

        # Define marker styles for different plot types to distinguish them
        marker_styles = ['o', 's', '^', 'D']

        # Handle line styles based on overlay_line_styles parameter
        if overlay_line_styles is None:
            # Auto-assign different styles for distinction
            line_styles = ['-', '--', '-.', ':']
        elif isinstance(overlay_line_styles, str):
            # Use same style for all plots
            line_styles = [overlay_line_styles] * n_plots
        elif isinstance(overlay_line_styles, list):
            # Use provided list of styles
            if len(overlay_line_styles) < n_plots:
                # Extend with default if not enough styles provided
                line_styles = overlay_line_styles + ['-'] * (n_plots - len(overlay_line_styles))
            else:
                line_styles = overlay_line_styles
        else:
            raise ValueError("overlay_line_styles must be a string, list, or None")

        # Create legend elements list
        legend_elements = []

        # Plot all types on the same axes
        for idx, ptype in enumerate(plot_types):
            config = plot_configs[ptype]
            y_col_millions = f"{config['y_col']}_millions"

            # Use different marker and line styles for each plot type
            current_marker = marker_styles[idx % len(marker_styles)]
            current_line = line_styles[idx % len(line_styles)]

            # Plot the Pareto front
            line_handle = ax.plot(df['eens_gwh'], df[y_col_millions],
                                  current_line, linewidth=line_width,
                                  color=config['color'], alpha=0.7,
                                  label=config.get('ylabel', ptype).replace(' (£ Million)', ''))[0]

            # Add markers
            ax.scatter(df['eens_gwh'], df[y_col_millions],
                       s=marker_size, marker=current_marker,
                       color=config['color'], edgecolors='black',
                       linewidth=1.5, zorder=5)

            legend_elements.append(line_handle)

            # Add threshold labels only for the first plot type to avoid clutter
            if show_threshold_labels and idx == 0:
                for i in range(0, len(df), annotation_spacing):
                    row = df.iloc[i]
                    threshold_val = row[threshold_col]

                    # Format threshold value
                    if threshold_val == 'Inf':
                        label_text = 'infinite'
                    elif threshold_val >= 1000:
                        label_text = f'{threshold_val / 1000:.1f}' if (
                                                                                  threshold_val / 1000) % 1 else f'{int(threshold_val / 1000)}'
                    else:
                        label_text = f'{threshold_val:.0f}'

                    # Position annotations
                    offset_x = 0.15
                    offset_y = df[y_col_millions].max() * 0.02
                    if i % 2 == 0:
                        offset_y = -offset_y

                    ax.annotate(label_text,
                                xy=(row['eens_gwh'], row[y_col_millions]),
                                xytext=(row['eens_gwh'] + offset_x, row[y_col_millions] + offset_y),
                                fontsize=annotation_fontsize,
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='white',
                                          edgecolor='gray',
                                          alpha=0.8),
                                arrowprops=dict(arrowstyle='->',
                                                connectionstyle='arc3,rad=0.2',
                                                color='gray',
                                                alpha=0.6,
                                                lw=0.5))

        # Set labels and title
        ax.set_xlabel('EENS at distribution level across all windstorm scenarios (GWh)',
                      fontsize=label_fontsize)
        ax.set_ylabel('Investment Cost (£ Million)', fontsize=label_fontsize)

        # Set title - use custom title if provided as string, otherwise default
        if isinstance(custom_titles, str):
            ax.set_title(custom_titles, fontsize=title_fontsize, fontweight='bold', pad=15)
        else:
            ax.set_title('Pareto Front: Multiple Investment Costs vs Resilience Metric Trade-off',
                         fontsize=title_fontsize, fontweight='bold', pad=15)

        # Set tick font sizes
        ax.tick_params(axis='both', labelsize=tick_fontsize)

        # Add grid
        if show_grid:
            ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.5)

        # Add legend
        ax.legend(handles=legend_elements, loc='upper right',
                  fontsize=legend_fontsize, frameon=True, shadow=True)

        # Set axis limits with padding
        x_padding = (df['eens_gwh'].max() - df['eens_gwh'].min()) * 0.05
        all_y_cols = [f"{config['y_col']}_millions" for config in plot_configs.values()
                      if config['y_col'] in df.columns]
        y_max = max(df[col].max() for col in all_y_cols if col in df.columns)
        y_min = min(df[col].min() for col in all_y_cols if col in df.columns)
        y_padding = (y_max - y_min) * 0.05

        ax.set_xlim(df['eens_gwh'].min() - x_padding, df['eens_gwh'].max() + x_padding)
        ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)

        axes_to_return = ax

    elif n_plots == 1:
        # Single plot
        fig, ax = plt.subplots(figsize=figure_size)
        axes = [ax]
        axes_to_return = ax
    elif n_plots == 2:
        # Two plots side by side
        fig, axes = plt.subplots(1, 2, figsize=(figure_size[0] * 1.8, figure_size[1]))
        axes_to_return = axes
    elif n_plots == 3:
        # Three plots in a row
        fig, axes = plt.subplots(1, 3, figsize=(figure_size[0] * 2.5, figure_size[1]))
        axes_to_return = axes
    elif n_plots == 4:
        # Four plots in 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(figure_size[0] * 1.5, figure_size[1] * 1.5))
        axes = axes.flatten()
        axes_to_return = axes
    else:
        raise ValueError(f"Too many plot types requested ({n_plots}). Maximum is 4.")

    # Create individual plots if not overlaying
    if not (overlay_plots and n_plots > 1):
        # Ensure axes is always a list
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        # Create plots
        for idx, ptype in enumerate(plot_types):
            ax = axes[idx]
            config = plot_configs[ptype]
        y_col_millions = f"{config['y_col']}_millions"

        # Plot the Pareto front
        ax.plot(df['eens_gwh'], df[y_col_millions],
                line_style, linewidth=line_width,
                color=config['color'], alpha=0.7, label='Pareto Front')

        # Add markers
        ax.scatter(df['eens_gwh'], df[y_col_millions],
                   s=marker_size, marker=marker_style,
                   color=config['color'], edgecolors='darkgreen',
                   linewidth=1.5, zorder=5)

        # Add threshold labels if requested
        if show_threshold_labels:
            for i in range(0, len(df), annotation_spacing):
                row = df.iloc[i]
                threshold_val = row[threshold_col]

                # Format threshold value based on its magnitude
                if threshold_val == 'Inf':
                    label_text = 'infinite'
                elif threshold_val >= 1000:
                    label_text = f'{threshold_val / 1000:.1f}' if (
                                                                              threshold_val / 1000) % 1 else f'{int(threshold_val / 1000)}'
                else:
                    label_text = f'{threshold_val:.0f}'

                # Position annotations with slight offset to avoid overlap
                offset_x = 0.15  # GWh offset
                offset_y = df[y_col_millions].max() * 0.02  # 2% of y-range offset

                # Alternate offset direction for better readability
                if i % 2 == 0:
                    offset_y = -offset_y

                ax.annotate(label_text,
                            xy=(row['eens_gwh'], row[y_col_millions]),
                            xytext=(row['eens_gwh'] + offset_x, row[y_col_millions] + offset_y),
                            fontsize=annotation_fontsize,
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white',
                                      edgecolor='gray',
                                      alpha=0.8),
                            arrowprops=dict(arrowstyle='->',
                                            connectionstyle='arc3,rad=0.2',
                                            color='gray',
                                            alpha=0.6,
                                            lw=0.5))

        # Set labels and title
        ax.set_xlabel('EENS at distribution level across all windstorm scenarios (GWh)',
                      fontsize=label_fontsize)
        ax.set_ylabel(config['ylabel'], fontsize=label_fontsize)
        ax.set_title(config['title'], fontsize=title_fontsize, fontweight='bold', pad=15)

        # Set tick font sizes
        ax.tick_params(axis='both', labelsize=tick_fontsize)

        # Add grid
        if show_grid:
            ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.5)

        # Add legend with threshold explanation
        if show_threshold_labels:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=config['color'], lw=line_width,
                       label='Pareto Front'),
                Line2D([0], [0], marker='s', color='w',
                       markerfacecolor='white', markeredgecolor='gray',
                       markersize=8, markeredgewidth=1,
                       label='EENS Threshold (GWh)')
            ]
            ax.legend(handles=legend_elements, loc='upper right',
                      fontsize=legend_fontsize, frameon=True, shadow=True)

            # Set axis limits with some padding
            x_padding = (df['eens_gwh'].max() - df['eens_gwh'].min()) * 0.05
            y_padding = (df[y_col_millions].max() - df[y_col_millions].min()) * 0.05
            ax.set_xlim(df['eens_gwh'].min() - x_padding, df['eens_gwh'].max() + x_padding)
            ax.set_ylim(max(0, df[y_col_millions].min() - y_padding),
                        df[y_col_millions].max() + y_padding)

    # Adjust layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    # Show plot
    plt.show()

    return fig, axes_to_return


# Example usage function (can be called from main.py or other scripts)
def generate_pareto_analysis_plots(hrdn_shift_value=20, plot_types=None, custom_titles=None):
    """
    Generate Pareto front analysis plots for a given hardening shift value.

    Parameters
    ----------
    hrdn_shift_value : int
        The hardening shift value used in the filename.
    plot_types : list of str or None
        List of plot types to generate. If None, generates all four types.
    custom_titles : dict or None
        Custom titles for each plot type.
    """
    import os

    # Create output directory if it doesn't exist
    output_dir = "Images_and_Plots/pareto_fronts"
    os.makedirs(output_dir, exist_ok=True)

    # Default to all plot types if not specified
    if plot_types is None:
        plot_types = ['total_investment', 'line_hardening', 'dg_installation', 'ess_installation']

    # Generate individual plots
    for ptype in plot_types:
        save_path = os.path.join(output_dir, f"pareto_front_{ptype}_hrdn_{hrdn_shift_value}.png")

        # Use custom title if provided
        title = custom_titles.get(ptype) if custom_titles else None

        fig, ax = plot_pareto_front_from_excel(
            excel_path=f"Post-processed_Data_for_Plots/investment_costs_vs_resilience_metrics_at_{hrdn_shift_value}_hrdn_shift.xlsx",
            plot_types=ptype,
            custom_titles=title,
            show_threshold_labels=True,
            annotation_spacing=2,  # Show every 2nd label to avoid crowding
            figure_size=(12, 8),
            save_path=save_path,
            show_grid=True
        )
        print(f"Generated {ptype} plot")

    # Option to generate combined plot with all requested types
    if len(plot_types) > 1:
        save_path_combined = os.path.join(output_dir, f"pareto_front_combined_hrdn_{hrdn_shift_value}.png")
        fig, axes = plot_pareto_front_from_excel(
            excel_path=f"Post-processed_Data_for_Plots/investment_costs_vs_resilience_metrics_at_{hrdn_shift_value}_hrdn_shift.xlsx",
            plot_types=plot_types,
            custom_titles=custom_titles,
            show_threshold_labels=True,
            annotation_spacing=3,  # Show every 3rd label in combined plot
            figure_size=(14, 10) if len(plot_types) > 2 else (14, 6),
            save_path=save_path_combined,
            show_grid=True
        )
        print(f"Generated combined plot with {len(plot_types)} subplots")

    return "Selected Pareto front plots generated successfully!"


# Usage Examples
"""
Example 1: Plot a single graph with default title
-------------------------------------------------
fig, ax = plot_pareto_front_from_excel(
    plot_types='total_investment'
)

Example 2: Plot multiple graphs overlaid with all solid lines
-------------------------------------------------------------
fig, ax = plot_pareto_front_from_excel(
    plot_types=['total_investment', 'line_hardening', 'dg_installation', 'ess_installation'],
    overlay_plots=True,
    overlay_line_styles='-',  # All lines will be solid
    custom_titles='Multi-Investment Strategy Comparison',
    figure_size=(14, 8)
)

Example 3: Overlay with custom line styles for each plot
--------------------------------------------------------
fig, ax = plot_pareto_front_from_excel(
    plot_types=['total_investment', 'line_hardening', 'dg_installation'],
    overlay_plots=True,
    overlay_line_styles=['-', '--', '-.'],  # Solid, dashed, dash-dot
    custom_titles='Comparison of Investment Strategies',
    figure_size=(14, 8)
)

Example 4: Plot specific graphs in separate subplots
----------------------------------------------------
fig, axes = plot_pareto_front_from_excel(
    plot_types=['total_investment', 'line_hardening'],
    overlay_plots=False,  # This is the default
    custom_titles={
        'total_investment': 'Investment vs Resilience Trade-off (20m/s hardening)',
        'line_hardening': 'Line Hardening Strategy Analysis'
    }
)

Example 5: Overlay with automatic line style differentiation (default)
----------------------------------------------------------------------
fig, ax = plot_pareto_front_from_excel(
    plot_types=['total_investment', 'line_hardening', 'dg_installation', 'ess_installation'],
    overlay_plots=True,
    # overlay_line_styles=None,  # This is default - auto-assigns different styles
    custom_titles='Multi-Investment Strategy Comparison',
    show_threshold_labels=True,
    annotation_spacing=2,
    figure_size=(12, 8),
    save_path='Images_and_Plots/overlaid_costs_comparison.png'
)

Example 6: Single plot with all customizations
----------------------------------------------
fig, ax = plot_pareto_front_from_excel(
    plot_types='line_hardening',
    custom_titles='Optimal Line Hardening Strategy under Windstorm Scenarios',
    show_threshold_labels=True,
    annotation_spacing=1,  # Show all labels
    title_fontsize=16,
    label_fontsize=14,
    tick_fontsize=12,
    annotation_fontsize=10,
    marker_size=150,
    line_style='--',
    save_path='Images_and_Plots/line_hardening_analysis.png'
)

Example 7: Compare two strategies with both using solid lines
-------------------------------------------------------------
fig, ax = plot_pareto_front_from_excel(
    plot_types=['line_hardening', 'dg_installation'],
    overlay_plots=True,
    overlay_line_styles='-',  # Both will use solid lines
    custom_titles='Line Hardening vs DG Installation Trade-offs',
    show_threshold_labels=True,
    figure_size=(12, 8)
)

Example 8: Using the batch generation helper
--------------------------------------------
# Generate only specific plots
generate_pareto_analysis_plots(
    hrdn_shift_value=20,
    plot_types=['total_investment', 'line_hardening'],
    custom_titles={
        'total_investment': 'Total Cost Analysis (20m/s)',
        'line_hardening': 'Hardening Strategy (20m/s)'
    }
)
"""