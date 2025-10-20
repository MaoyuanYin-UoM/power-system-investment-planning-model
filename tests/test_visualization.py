from visualizations.visualization import *
from visualizations.visualize_pareto_fronts import plot_pareto_front_from_excel

visualize_fragility_curve(mu=3.42, sigma=0.13, thrd_1=20, thrd_2=50, shift_f=0,
                          title='Fragility Curve for TN OHL')  # in m/s

visualize_fragility_curve(mu=3.18, sigma=0.14, thrd_1=10, thrd_2=40, shift_f=0,
                         title='Fragility Curve for DN OHL')  # in m/s

# visualize_bch_and_ws_contour(
#                              # network_name="GB_Transmission_Network_29_Bus",
#                              network_name="29_bus_GB_transmission_network_with_Kearsley_GSP_group",
#                              windstorm_name="windstorm_29_bus_GB_transmission_network",
#                              zoomed_distribution=False,
#                              zoom_border=0.2,
#                              label_buses=False,
#                              )

# fig, ax = plot_pareto_front_from_excel(
#     excel_path="Post-processed_Data_for_Plots/ws_sce1000_seed10000_k4_(updated_frg_curve)/investment_costs_vs_resilience_metrics_20_hrdn_shift_70-110_vmax_20-60_vmin.xlsx",
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

