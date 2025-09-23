"""
Windstorm-event visualization utilities.

This module focuses ONLY on plotting windstorm events from a generated
scenario library JSON, keeping windstorm-specific plotting separate
from general-purpose visualization helpers.

It supports both the “old” list-based libraries and the newer dict-based
libraries created by ws_scenario_library_generator (keyed by 'ws_XXXX').
"""

import json
import math
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib import cm

from factories.network_factory import make_network
from factories.windstorm_factory import make_windstorm


# -----------------------------
# Helpers
# -----------------------------
def km_to_deg_radius(lat_deg: float, r_km: float) -> float:
    """Convert radius in km to degrees of latitude at given latitude (ok for small circles)."""
    return r_km / 111.0  # use latitude scale; conservative on longitude


def segment_capsule_polygon(p0: Tuple[float, float],
                            p1: Tuple[float, float],
                            r_deg: float,
                            cap_samples: int = 12) -> np.ndarray:
    """
    Build a capsule (rounded rectangle) polygon around segment p0->p1 with radius r_deg.
    Returns Nx2 array of (lon, lat) vertices.
    """
    x0, y0 = p0
    x1, y1 = p1
    vx, vy = x1 - x0, y1 - y0
    seg_len = math.hypot(vx, vy)
    if seg_len == 0:
        # Degenerate: return a circle
        theta = np.linspace(0, 2 * np.pi, 36, endpoint=True)
        return np.c_[x0 + r_deg * np.cos(theta), y0 + r_deg * np.sin(theta)]

    # Unit perpendicular (to the left)
    nx, ny = -vy / seg_len, vx / seg_len

    # Offset corners
    a1 = (x0 + nx * r_deg, y0 + ny * r_deg)
    a2 = (x1 + nx * r_deg, y1 + ny * r_deg)
    b2 = (x1 - nx * r_deg, y1 - ny * r_deg)
    b1 = (x0 - nx * r_deg, y0 - ny * r_deg)

    # Caps: semicircles at p1 and p0
    th1 = math.atan2(ny, nx)            # normal direction angle
    th_seg = math.atan2(vy, vx)         # segment direction angle

    # End cap around p1 (from +normal to -normal, sweeping forward)
    theta_end = np.linspace(th_seg - np.pi/2, th_seg + np.pi/2, cap_samples)
    cap_end = np.c_[x1 + r_deg * np.cos(theta_end), y1 + r_deg * np.sin(theta_end)]

    # Start cap around p0 (from -normal to +normal, sweeping backward)
    theta_start = np.linspace(th_seg + np.pi/2, th_seg - np.pi/2, cap_samples)
    cap_start = np.c_[x0 + r_deg * np.cos(theta_start), y0 + r_deg * np.sin(theta_start)]

    # Assemble polygon (CCW): a1 -> a2 -> end cap -> b2 -> b1 -> start cap
    poly = np.vstack([
        np.array(a1),
        np.array(a2),
        cap_end,
        np.array(b2),
        np.array(b1),
        cap_start
    ])
    return poly


def draw_network(ax, net, tn_color="darkgreen", dn_color="orange", lw=1.2, alpha=0.9):
    """Plot network branches with separate colours for T/D."""
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
            label = "Transmission Branch" if not tn_plotted else None
            tn_plotted = True
        else:
            color = dn_color
            label = "Distribution Branch" if not dn_plotted else None
            dn_plotted = True

        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, alpha=alpha, label=label)


def _auto_marker_stride(n_points: int, target_markers: int = 22) -> int:
    """Return markevery stride so we show about `target_markers` markers along a path."""
    if n_points <= target_markers:
        return 1
    return max(1, n_points // target_markers)


# -----------------------------
# Core plotting
# -----------------------------
def plot_ws_scenarios(
    ws_library_path: str,
    show_network: bool = True,
    plot_circles: bool = True,
    plot_envelopes: bool = True,
    multi_on_one_axis: bool = True,
    show_per_scenario_legend: bool = False,
    figsize: Tuple[int, int] = (11, 9),
    title: Optional[str] = None,
    circle_alpha: float = 0.18,
    envelope_alpha: float = 0.20,
    path_alpha: float = 0.9,
    title_fontsize: int = 15,          # New parameter for title font size
    xlabel_fontsize: int = 12,         # New parameter for x-axis label font size
    ylabel_fontsize: int = 12,         # New parameter for y-axis label font size
    tick_fontsize: int = 10,           # New parameter for tick font size
    legend_fontsize: int = 10,         # New parameter for legend font size
):
    """
    Tidy visualization of all scenarios in a library.

    - Generic legend items only (unless `show_per_scenario_legend=True`).
    - “Storm radius” legend entry added with a circle marker.
    - Envelope shares the same hue as the path, with high transparency.
    """
    with open(ws_library_path, "r") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    scenarios: Dict[str, Any] = data.get("scenarios") or data.get("ws_scenarios")

    # Extract scenario probabilities if available
    scenario_probabilities = data.get("scenario_probabilities", {})

    if isinstance(scenarios, list):
        # Old format → list; give IDs
        scenarios = {f"ws_{i:04d}": sc for i, sc in enumerate(scenarios)}
    scenario_ids = list(scenarios.keys())

    # Network & windstorm objects (for axis extents + consistent style)
    net = make_network(meta.get("network_preset", "29_bus_GB_transmission_network_with_Kearsley_GSP_group"))
    ws = make_windstorm(meta.get("windstorm_preset", "windstorm_GB_transmission_network"))

    # Figure/axis
    fig, ax = plt.subplots(figsize=figsize)

    # Network underlay
    if show_network:
        draw_network(ax, net)

    # Colors for scenarios - avoid green and orange which are used for TN/DN branches
    # tab10 colormap indices: 0=blue, 1=orange, 2=green, 3=red, 4=purple, 5=brown, 6=pink, 7=gray, 8=olive, 9=cyan
    # Skip indices 1 (orange) and 2 (green) to avoid conflicts with network colors
    cmap = cm.get_cmap("tab10")
    safe_indices = [0, 1, 4, 5, 6, 7, 8, 9]  # Exclude 2, 3
    colors = [cmap(safe_indices[i % len(safe_indices)]) for i in range(len(scenario_ids))]

    # Plot each scenario
    per_scn_handles = []
    for idx, scn_id in enumerate(scenario_ids):
        scn = scenarios[scn_id]
        # Each scenario may have multiple events → draw them all in the same color
        sc_color = colors[idx]

        for ev in scn.get("events", []):
            epi = np.asarray(ev["epicentre"], dtype=float)  # shape (T, 2)
            r_km = ev["radius"]
            if isinstance(r_km, (int, float)):
                r_km = [r_km] * len(epi)
            r_deg = [km_to_deg_radius(lat, rk) for (lon, lat), rk in zip(epi, r_km)]

            # Path
            stride = _auto_marker_stride(len(epi))
            (ln,) = ax.plot(
                epi[:, 0],
                epi[:, 1],
                "-o",
                lw=1.8,
                ms=4.5,
                markevery=stride,
                color=sc_color,
                alpha=path_alpha,
                label=None,
                zorder=3,
            )

            # Circles (per-hour radius)
            if plot_circles:
                patches = []
                for (lon, lat), rr in zip(epi, r_deg):
                    patches.append(Circle((lon, lat), rr))
                pc = PatchCollection(patches, facecolor=sc_color, edgecolor="none", alpha=circle_alpha, zorder=2)
                ax.add_collection(pc)

            # Envelope (capsule per segment)
            if plot_envelopes and len(epi) > 1:
                env_patches = []
                for t in range(len(epi) - 1):
                    p0 = (epi[t, 0], epi[t, 1])
                    p1 = (epi[t + 1, 0], epi[t + 1, 1])
                    rr = r_deg[t]  # radius for hour t
                    poly = segment_capsule_polygon(p0, p1, rr, cap_samples=18)
                    # Build a closed Path
                    verts = np.vstack([poly, poly[0]])
                    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(poly) - 1) + [MplPath.CLOSEPOLY]
                    path = MplPath(verts, codes)
                    env_patches.append(PathPatch(path))

                env = PatchCollection(
                    env_patches,
                    facecolor=sc_color,
                    edgecolor="none",
                    alpha=envelope_alpha,
                    zorder=2,
                )
                ax.add_collection(env)

        # Optionally add a legend entry per scenario (kept off by default)
        if show_per_scenario_legend:
            # per_scn_handles.append(Line2D([0], [0], color=sc_color, lw=2, label=f"{scn_id} (paths/envelopes)"))

            # Create legend entry with scenario probability
            # Extract scenario number from ID (e.g., "ws_0001" -> 1)
            scenario_num = idx + 1  # Simple 1-based numbering
            # Get probability for this scenario (default to equal probability if not found)
            prob = scenario_probabilities.get(scn_id, 1.0 / len(scenario_ids))

            # Add legend handle with scenario number and probability
            legend_label = f"Scenario {scenario_num}, p = {prob:.3f}"
            per_scn_handles.append(Line2D([0], [0], color=sc_color, lw=2, label=legend_label))

    # Axis limits: pad around network
    bus_lons = net._get_bus_lon()
    bus_lats = net._get_bus_lat()
    xmin, xmax = min(bus_lons), max(bus_lons)
    ymin, ymax = min(bus_lats), max(bus_lats)
    pad_x = 0.7
    pad_y = 0.7
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    # Titles & labels
    if title is None:
        if len(scenario_ids) == 1:
            lbl = scenario_ids[0]
            ax.set_title(f"Windstorm Path – {lbl}", fontsize=title_fontsize, fontweight="bold")
        else:
            ax.set_title("Windstorm Paths – Multiple Scenarios", fontsize=title_fontsize, fontweight="bold")
    else:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    ax.set_xlabel("Longitude", fontsize=xlabel_fontsize)
    ax.set_ylabel("Latitude", fontsize=ylabel_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)

    # -----------------------------
    # Legend
    # -----------------------------
    legend_handles: List[Any] = []

    # Network entries (already plotted with labels in draw_network)
    # Extract unique labelled handles actually present
    cur_handles, cur_labels = ax.get_legend_handles_labels()
    seen = set()
    for h, lab in zip(cur_handles, cur_labels):
        if lab and lab not in seen and lab in ("Transmission Branch", "Distribution Branch"):
            legend_handles.append(Line2D([0], [0], color=h.get_color(), lw=2, label=lab))
            seen.add(lab)

    # 1) Windstorm Path (generic)
    legend_handles.append(Line2D([0], [0], color="#1f77b4", lw=2, marker="o", label="Windstorm Path"))

    # 2) Storm radius (as a circle marker, filled)
    legend_handles.append(Line2D([0], [0], linestyle="None", marker="o", ms=9,
                                 markerfacecolor="#1f77b4", markeredgecolor="none",
                                 alpha=0.35, label="Storm radius"))

    # 3) Storm Envelope (filled patch)
    legend_handles.append(FancyBboxPatch((0, 0), 1, 1,
                                         boxstyle="round,pad=0.0",
                                         linewidth=0,
                                         facecolor="#1f77b4",
                                         alpha=0.22,
                                         label="Storm Envelope"))

    # 4） Per scenario data
    legend_handles.extend(per_scn_handles)

    ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=legend_fontsize)


    plt.tight_layout()
    plt.show()
    return fig, ax


# -----------------------------
# Quick single-event helper (kept simple)
# -----------------------------
def plot_single_event(ws_library_path: str,
                      scenario_id: str,
                      event_idx: int = 0,
                      **kwargs):
    """Convenience wrapper to plot one chosen scenario/event with the same clean legend."""
    with open(ws_library_path, "r") as f:
        data = json.load(f)
    scenarios: Dict[str, Any] = data.get("scenarios") or data.get("ws_scenarios")
    if isinstance(scenarios, list):
        scenarios = {f"ws_{i:04d}": sc for i, sc in enumerate(scenarios)}
    # Slice a small dict containing just one scenario
    mini = {"metadata": data.get("metadata", {}), "scenarios": {scenario_id: scenarios[scenario_id]}}
    tmp = Path(".") / "_tmp_single_ws.json"
    with open(tmp, "w") as f:
        json.dump(mini, f)
    try:
        return plot_ws_scenarios(str(tmp), **kwargs)
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


# -----------------------------
# Script entry
# -----------------------------
if __name__ == "__main__":
    # Example: plot 10 scenarios from a filtered library on one canvas.
    # Adjust the path below to your library.
    ws_library = "../Scenario_Database/Scenarios_Libraries/Clustered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_29GB_5000scn_s10000_filt_b1_h1_buf15_eens_k10.json"

    plot_ws_scenarios(
        ws_library_path=ws_library,
        show_network=True,
        plot_circles=True,  # circles visible (constant radius → feel free to set False)
        plot_envelopes=True,  # capsule envelope
        multi_on_one_axis=True,  # plot all scenarios together
        show_per_scenario_legend=True,  # keep legend compact
        figsize=(10, 10),
        title="Windstorm Scenarios Visualisation",
        # Font size parameters (optional - will use defaults if not specified)
        title_fontsize=16,  # Font size for title
        xlabel_fontsize=14,  # Font size for x-axis label
        ylabel_fontsize=14,  # Font size for y-axis label
        tick_fontsize=12,  # Font size for axis ticks
        legend_fontsize=13,  # Font size for legend
    )