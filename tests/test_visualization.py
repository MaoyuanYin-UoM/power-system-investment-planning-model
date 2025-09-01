from visualizations.visualization import *
from visualizations.visualize_pareto_fronts import plot_pareto_front_from_excel

visualize_bch_and_ws_contour(
                             # network_name="GB_Transmission_Network_29_Bus",
                             network_name="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
                             windstorm_name="windstorm_29_bus_GB_transmission_network",
                             zoomed_distribution=True,
                             zoom_border=0.2,
                             label_buses=False,
                             )

# fig, ax = plot_pareto_front_from_excel(
#     excel_path="Post-processed_Data_for_Plots/ws_sce1000_seed20000_k6/investment_costs_vs_resilience_metrics_30_hrdn_shift_70-90_vmax_45-55_vmin.xlsx",
#     # excel_path="Post-processed_Data_for_Plots/ws_sce20_seed10000/investment_costs_vs_resilience_metrics_at_30_hrdn_shift.xlsx",
#     plot_types=['total_investment', 'line_hardening', 'dg_installation', 'ess_installation'],
#     overlay_plots=True,
#     overlay_line_styles='-',
#     custom_titles="Pareto Front under Severe Windstorms at 30mph Hardening Shift",
#     show_threshold_labels=False,
#     title_fontsize=19,
#     label_fontsize=18,
#     tick_fontsize=16,
#     annotation_fontsize=14,
#     legend_fontsize=16,
#     line_width=3,
#     marker_size=150,
#     figure_size=(12, 8)
# )

