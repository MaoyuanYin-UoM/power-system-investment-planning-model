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

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Patch

from factories.network_factory import make_network  # uses metadata.network_preset


# ======================= USER SETTINGS =======================


LIBRARY_PATH = "../Scenario_Database/Scenarios_Libraries/Original_Scenario_Libraries/windstorm_library_net_29BusGB-KearsleyGSP_ws_29BusGB_10scenarios_seed50000.json"
# LIBRARY_PATH = "../Scenario_Database/Scenarios_Libraries/Clustered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_29GB_1000scn_s50000_filt_b1_h1_buf15_eens_k10.json"

# Choose a single scenario by id (e.g., "ws_0003"), by 1-based index (e.g., 3), or None for the first
SCENARIO_ID_OR_INDEX: Optional[object] = None

# ---- Appearance / fonts
FIGSIZE = (10, 8)
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10

# ---- Event path / disks
PLOT_CIRCLES = True
PATH_ALPHA = 0.9
MARKER_SIZE = 40
DISK_ALPHA = 0.20

# ---- Continuous-motion visuals (envelopes)
# Pairwise swept band between consecutive epicentres (recommended)
PLOT_PAIRWISE_SWEEP = True
# Per-circle circumbox (square 2r) — alternative diagnostic view
PLOT_PER_CIRCLE_BOX = False

# Envelope styling (by default uses the *same* colour as the path but lighter)
SWEEP_FACE_ALPHA = 0.20
SWEEP_EDGE_ALPHA = 0.9

# ======================= LIB HELPERS ========================

def _load_library(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _iter_scenarios(lib: Dict[str, Any]):
    """
    Iterate scenarios in a consistent way for both dict and list formats.
    Yields tuples: (scenario_id, scenario_data)
    """
    if "scenarios" in lib:
        scenarios = lib["scenarios"]
    elif "ws_scenarios" in lib:  # legacy
        scenarios = lib["ws_scenarios"]
    else:
        raise ValueError("No scenarios found. Expected 'scenarios' or 'ws_scenarios'.")

    if isinstance(scenarios, dict):
        def key_fn(sid: str):
            try:
                return (int(sid.split("_")[-1]), sid)
            except Exception:
                return (10**9, sid)
        for sid in sorted(scenarios.keys(), key=key_fn):
            yield sid, scenarios[sid]
    elif isinstance(scenarios, list):
        for i, s in enumerate(scenarios):
            sid = s.get("scenario_id", f"ws_{i:04d}")
            yield sid, s
    else:
        raise TypeError("Unsupported scenarios container type.")


def _pick_scenario(lib: Dict[str, Any], selector: Optional[object]) -> Tuple[str, Dict[str, Any]]:
    all_scenarios = list(_iter_scenarios(lib))
    if not all_scenarios:
        raise ValueError("Library has no scenarios.")

    if selector is None:
        return all_scenarios[0]

    if isinstance(selector, int):  # 1-based
        if selector < 1 or selector > len(all_scenarios):
            raise IndexError(f"Scenario index {selector} out of range (1..{len(all_scenarios)}).")
        return all_scenarios[selector - 1]

    for sid, sdata in all_scenarios:
        if sid == selector:
            return sid, sdata

    raise KeyError(f"Scenario '{selector}' not found. First few: {[sid for sid,_ in all_scenarios[:6]]}")


def _km_to_deg_lat(km: float) -> float:
    """Approx convert km to degrees of latitude (1° ≈ 111 km)."""
    return km / 111.0


# ======================= GEOMETRY HELPERS ========================

def _unit_perp(p1, p2):
    """Unit vector perpendicular to segment p1->p2 (in degrees)."""
    vx = p2[0] - p1[0]
    vy = p2[1] - p1[1]
    nx, ny = -vy, vx
    nrm = math.hypot(nx, ny)
    if nrm == 0:
        return (0.0, 0.0)
    return (nx / nrm, ny / nrm)


def _rect_from_center_dirs(cx, cy, half_len, half_wid, tx, ty, nx, ny):
    """Oriented rectangle centred at (cx,cy) with tangent t=(tx,ty) and normal n=(nx,ny)."""
    p1 = (cx - half_len * tx - half_wid * nx, cy - half_len * ty - half_wid * ny)
    p2 = (cx + half_len * tx - half_wid * nx, cy + half_len * ty - half_wid * ny)
    p3 = (cx + half_len * tx + half_wid * nx, cy + half_len * ty + half_wid * ny)
    p4 = (cx - half_len * tx + half_wid * nx, cy - half_len * ty + half_wid * ny)
    return [p1, p2, p3, p4]


# ======================= NETWORK BACKGROUND ========================

def _draw_network_background(ax, network_preset: str):
    """Draw branches with T/D colouring; set axis to bus extents."""
    net = make_network(network_preset)
    net.set_gis_data()

    bgn = net._get_bch_gis_bgn()
    end = net._get_bch_gis_end()

    bus_lons = net._get_bus_lon()
    bus_lats = net._get_bus_lat()

    has_branch_levels = hasattr(net.data.net, 'branch_level')

    tn_plotted = False
    dn_plotted = False

    for idx, (p, q) in enumerate(zip(bgn, end), start=1):
        if has_branch_levels:
            level = net.data.net.branch_level.get(idx, 'T')
        else:
            level = 'T'

        if level in ('T', 'T-D'):
            color = 'darkgreen'
            label = 'Transmission Branch' if not tn_plotted else ""
            tn_plotted = True
        else:
            color = 'orange'
            label = 'Distribution Branch' if not dn_plotted else ""
            dn_plotted = True

        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, alpha=0.7, label=label)

    xmin, xmax = min(bus_lons), max(bus_lons)
    ymin, ymax = min(bus_lats), max(bus_lats)
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)

    return net


# ======================= EVENT PLOTTING ========================

def _draw_pairwise_sweep_band(ax, epicentre: List[List[float]], radii_km: List[float], color: str):
    """
    For each consecutive pair (t, t+1), draw a band equal to the Minkowski sum
    of the line segment with a disk of radius min(r_t, r_{t+1}).
    End circles (already plotted) cover larger radius differences.
    """
    T = len(epicentre)
    if T < 2:
        return

    # Make sure legend has exactly one "Storm Envelope" entry
    have_env_label = "Storm Envelope" in ax.get_legend_handles_labels()[1]
    env_label = None if have_env_label else "Storm Envelope"

    for t in range(T - 1):
        p1 = epicentre[t]
        p2 = epicentre[t + 1]
        r_deg = _km_to_deg_lat(min(radii_km[t], radii_km[t + 1]))

        nx, ny = _unit_perp(p1, p2)
        if nx == 0.0 and ny == 0.0:
            continue

        c1 = (p1[0] + nx * r_deg, p1[1] + ny * r_deg)
        c2 = (p2[0] + nx * r_deg, p2[1] + ny * r_deg)
        c3 = (p2[0] - nx * r_deg, p2[1] - ny * r_deg)
        c4 = (p1[0] - nx * r_deg, p1[1] - ny * r_deg)

        band = Polygon([c1, c2, c3, c4], closed=True,
                       facecolor=color, edgecolor=color,
                       alpha=SWEEP_FACE_ALPHA, zorder=2, label=env_label)
        ax.add_patch(band)

        # Only label the first band once
        env_label = None


def _draw_per_circle_boxes(ax, epicentre: List[List[float]], radii_km: List[float], color: str):
    """Draw a circumbox (square) of side 2r for each circle, oriented by the local tangent."""
    T = len(epicentre)
    if T == 0:
        return

    for t in range(T):
        cx, cy = epicentre[t]
        r_deg = _km_to_deg_lat(radii_km[t])

        # tangent from neighbours (forward/backward at ends)
        if t == 0:
            tx = (epicentre[1][0] - cx) if T > 1 else 1.0
            ty = (epicentre[1][1] - cy) if T > 1 else 0.0
        elif t == T - 1:
            tx = cx - epicentre[t - 1][0]
            ty = cy - epicentre[t - 1][1]
        else:
            tx = epicentre[t + 1][0] - epicentre[t - 1][0]
            ty = epicentre[t + 1][1] - epicentre[t - 1][1]

        nrm = math.hypot(tx, ty)
        if nrm == 0:
            tx, ty = 1.0, 0.0
        else:
            tx, ty = tx / nrm, ty / nrm

        nx, ny = -ty, tx  # normal
        # Rect corners for half_len=r_deg, half_wid=r_deg
        p1 = (cx - r_deg * tx - r_deg * nx, cy - r_deg * ty - r_deg * ny)
        p2 = (cx + r_deg * tx - r_deg * nx, cy + r_deg * ty - r_deg * ny)
        p3 = (cx + r_deg * tx + r_deg * nx, cy + r_deg * ty + r_deg * ny)
        p4 = (cx - r_deg * tx + r_deg * nx, cy - r_deg * ty + r_deg * ny)

        poly = Polygon([p1, p2, p3, p4], closed=True, fill=False,
                       edgecolor=color, alpha=0.8, lw=1.0)
        ax.add_patch(poly)


def _plot_event_over_network(ax, event: Dict[str, Any], color: str):
    """Overlay one windstorm event on an axis that already has the network drawn."""
    epic = event["epicentre"]             # [[lon, lat], ...]
    radii_km = event["radius"]            # [r_km, ...]

    # path + markers
    lons = [p[0] for p in epic]
    lats = [p[1] for p in epic]
    ax.plot(lons, lats, "o-", color=color, alpha=PATH_ALPHA, lw=1.5, label="Windstorm Path")
    ax.scatter(lons, lats, s=MARKER_SIZE, color=color, zorder=3)

    # hourly circles
    if PLOT_CIRCLES:
        for (lon, lat), r_km in zip(epic, radii_km):
            r_deg = _km_to_deg_lat(r_km)
            ax.add_patch(Circle((lon, lat), r_deg, color=color, alpha=DISK_ALPHA, fill=True))

    # continuous-motion visuals (envelope)
    if PLOT_PAIRWISE_SWEEP:
        _draw_pairwise_sweep_band(ax, epic, radii_km, color)

    if PLOT_PER_CIRCLE_BOX:
        _draw_per_circle_boxes(ax, epic, radii_km, color)


# ======================= DEFAULT: ONE FIGURE PER EVENT =======================

def _plot_scenario_events_with_network(sid: str, sdata: Dict[str, Any], network_preset: str, color: str = "blue"):
    events = sdata.get("events", [])
    if not events:
        print(f"[{sid}] No events to plot.")
        return

    for ev_idx, ev in enumerate(events, start=1):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        _draw_network_background(ax, network_preset)
        _plot_event_over_network(ax, ev, color=color)

        ax.set_xlabel("Longitude", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Latitude", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
        ax.set_title(f"Windstorm Path - {sid}, Event {ev_idx}", fontsize=TITLE_FONTSIZE, fontweight='bold')

        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        h_clean, l_clean = [], []
        for h, l in zip(handles, labels):
            if l and l not in seen:
                h_clean.append(h); l_clean.append(l); seen.add(l)
        if h_clean:
            ax.legend(h_clean, l_clean, fontsize=LEGEND_FONTSIZE, loc="best")

        ax.grid(True)
        plt.tight_layout()
        plt.show()


# ======================= OPTIONAL: MULTI-SCENARIO, SINGLE CANVAS =======================

# Toggle this to True to plot multiple scenarios on one canvas
MULTI_SCENARIO_SINGLE_CANVAS = True
# Leave None to plot ALL scenarios; or provide a list of ids/1-based indices, e.g. ["ws_0000", 2, 5]
SCENARIOS_TO_PLOT: Optional[List[object]] = None

def _color_cycle():
    for c in plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']):
        yield c

def _plot_multiple_scenarios_single_canvas(lib: Dict[str, Any], network_preset: str):
    scen_list = list(_iter_scenarios(lib))
    if SCENARIOS_TO_PLOT:
        # filter by id or 1-based index
        picked = []
        ids = {s for s in SCENARIOS_TO_PLOT if isinstance(s, str)}
        idxs = {s for s in SCENARIOS_TO_PLOT if isinstance(s, int)}
        for i, (sid, sdata) in enumerate(scen_list, start=1):
            if sid in ids or i in idxs:
                picked.append((sid, sdata))
        scen_list = picked

    if not scen_list:
        print("No scenarios selected to plot.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _draw_network_background(ax, network_preset)

    colour = _color_cycle()
    legend_added_for_path = set()

    for sid, sdata in scen_list:
        c = next(colour)
        for ev in sdata.get("events", []):
            # Draw event with this scenario's colour
            _plot_event_over_network(ax, ev, color=c)

        # Add one proxy legend handle per scenario for path colour
        if sid not in legend_added_for_path:
            ax.plot([], [], color=c, lw=2, label=f"{sid} (paths/envelopes)")
            legend_added_for_path.add(sid)

    ax.set_xlabel("Longitude", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Latitude", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.set_title("Windstorm Paths – Multiple Scenarios", fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h_clean, l_clean = [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            h_clean.append(h); l_clean.append(l); seen.add(l)
    if h_clean:
        ax.legend(h_clean, l_clean, fontsize=LEGEND_FONTSIZE, loc="best")

    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ======================= MAIN =======================

if __name__ == "__main__":
    lib = _load_library(LIBRARY_PATH)
    meta = lib.get("metadata", {})
    network_preset = meta.get("network_preset", None)
    if not network_preset:
        raise KeyError("metadata.network_preset missing in the library JSON; cannot draw network background.")

    if MULTI_SCENARIO_SINGLE_CANVAS:
        _plot_multiple_scenarios_single_canvas(lib, network_preset)
    else:
        sid, sdata = _pick_scenario(lib, SCENARIO_ID_OR_INDEX)
        # default “classic” behaviour: one figure per event (blue)
        _plot_scenario_events_with_network(sid, sdata, network_preset, color="C0")