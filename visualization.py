# This script contains function for visualization

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm


def visualize_ws_contour(WindConfig):
    """Visualize the starting- and ending-points contour for windstorm path generation"""
    wcon = WindConfig
    start_lon = wcon.data.WS.contour.start_lon
    start_lat = wcon.data.WS.contour.start_lat
    start_concty = wcon.data.WS.contour.start_connectivity
    end_lon = wcon.data.WS.contour.end_lon
    end_lat_coef = wcon.data.WS.contour.end_lat_coef

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


def visualize_bch_and_ws_contour(WindConfig, NetworkConfig):
    """
    Visualize the branches along with the starting- and ending-points contour for windstorm path generation.

    Parameters:
    - WindConfig: Configuration object containing windstorm data.
    - NetworkConfig: Configuration object containing network data.
    """
    # Windstorm data
    wcon = WindConfig
    start_lon = wcon.data.WS.contour.start_lon
    start_lat = wcon.data.WS.contour.start_lat
    start_concty = wcon.data.WS.contour.start_connectivity
    end_lon = wcon.data.WS.contour.end_lon
    end_lat_coef = wcon.data.WS.contour.end_lat_coef
    end_lat = [end_lat_coef[0] * x + end_lat_coef[1] for x in end_lon]

    # Branch data
    ncon = NetworkConfig
    bch_gis_bgn = ncon.data.net.bch_gis_bgn
    bch_gis_end = ncon.data.net.bch_gis_end

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

    # Labels, title, and legend
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Branches and Windstorm Contours")
    ax.legend()
    ax.grid(True)

    plt.show()