import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

# Font and general styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# Colors for domains (Modern accessible palette)
COLOR_A = "#0284c7"  # Deep Sky Blue
COLOR_B = "#d97706"  # Warm Amber
COLOR_C = "#059669"  # Emerald Green

BG_A = "#e0f2fe"
BG_B = "#fef3c7"
BG_C = "#d1fae5"

DOMAIN_COLORS = {'A': COLOR_A, 'B': COLOR_B, 'C': COLOR_C}
DOMAIN_BG = {'A': BG_A, 'B': BG_B, 'C': BG_C}

# =============================================================================
# 1. TOPOLOGY COORDINATES (Carefully tuned to avoid line crossovers)
# =============================================================================

# NSFNET (14 nodes)
# Positioned cleanly: West (A), Central (B), East (C)
# TX (7) is placed so Domain A forms ONE continuous unified shape without splitting
NSFNET_COORDS = {
    0: (14, 44),   # WA
    1: (12, 31),   # CA1
    2: (15, 17),   # CA2
    8: (25, 33),   # UT
    7: (34, 14),   # TX (continuous with Domain A)
    9: (38, 30),   # CO
    3: (48, 38),   # NE
    4: (58, 35),   # IL
    5: (64, 44),   # MI
    6: (69, 15),   # GA
    10: (75, 34),  # PA
    11: (83, 44),  # NY
    12: (88, 36),  # NJ
    13: (83, 25),  # DC
}

# NSFNET Curvatures to completely eliminate straight-line overlaps & cross-cutting
NSFNET_CURVATURES = {
    (0, 7): -0.25,   # WA <---> TX: curves outward to the west/south, clearing UT and CA
    (2, 5): -0.32,   # CA2 <---> MI: curves along the southern border, clearing CO/IL
    (8, 11): 0.25,   # UT <---> NY: curves gracefully over the north, clearing NE/IL/MI/PA
    (8, 12): -0.20,  # UT <---> NJ: curves slightly south
    (1, 3): 0.14,    # CA1 <---> NE: curves north of UT
    (5, 13): -0.18,  # MI <---> DC: curves south of PA
    (4, 6): 0.12,    # IL <---> GA: slight curve
    (6, 7): -0.12,   # GA <---> TX: slight curve along the south
}

# ABILENE (12 nodes)
ABILENE_COORDS = {
    10: (15, 45),  # STTL
    9: (13, 31),   # SNVA
    7: (17, 18),   # LOSA
    3: (32, 33),   # DNVR
    6: (47, 32),   # KSCY
    2: (57, 43),   # CHIN
    5: (63, 33),   # IPLS
    4: (48, 14),   # HSTN
    0: (72, 14),   # ATLAM5
    1: (75, 23),   # ATLA
    11: (82, 30),  # WASH
    8: (88, 42),   # NYCM
}

ABILENE_CURVATURES = {
    (3, 10): 0.20,   # DNVR <---> STTL: curves north, clearing SNVA
    (4, 7): -0.22,   # HSTN <---> LOSA: curves along the south, clearing everything
    (2, 8): 0.16,    # CHIN <---> NYCM: curves north, clearing WASH
    (1, 4): -0.15,   # ATLA <---> HSTN: curves gently along the south
}

# GÉANT (22 nodes)
GEANT_COORDS = {
    11: (13, 40),  # IE
    21: (22, 39),  # UK
    7: (26, 27),   # FR
    1: (32, 37),   # BE
    15: (35, 45),  # NL
    16: (39, 49),  # NO
    19: (48, 49),  # SE
    4: (43, 33),   # DE
    2: (36, 24),   # CH
    3: (52, 34),   # CZ
    0: (54, 25),   # AT
    17: (63, 42),  # PL
    10: (64, 25),  # HU
    20: (50, 16),  # SI
    9: (58, 15),   # HR
    18: (11, 14),  # PT
    6: (19, 16),   # ES
    14: (33, 20),  # LU
    13: (44, 18),  # IT
    5: (47, 42),   # DK
    8: (75, 14),   # GR
    12: (87, 12),  # IL
}

GEANT_CURVATURES = {
    (12, 13): -0.28,  # IL <---> IT: curves across the Mediterranean
    (17, 19): 0.22,   # PL <---> SE: curves over the Baltic
    (1, 21): 0.15,    # BE <---> UK: curves over the North Sea
    (7, 13): -0.18,   # FR <---> IT: curves south of Alps
    (4, 17): 0.15,    # DE <---> PL: slight curve
    (5, 16): 0.15,    # DK <---> NO: curves over water
}

# =============================================================================
# 2. HELPER TO DRAW A 100% UNIFIED, CONTINUOUS SMOOTH REGION
# =============================================================================
def draw_unified_domain_zone(ax, pts, border_color, fill_color, pad=4.8):
    """Draws a smooth, organic, 100% continuous closed region enclosing all pts."""
    pts = np.array(pts)
    if len(pts) < 3:
        # If 2 points, draw an ellipse/capsule
        p1, p2 = pts[0], pts[1]
        mid = (p1 + p2) / 2
        ax.scatter([mid[0]], [mid[1]], s=2500, c=fill_color, edgecolors=border_color, lw=2, ls='--', zorder=2)
        return

    # Convex hull ensures 100% CONNECTEDNESS (impossible to split into pieces)
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    
    # Expand hull vertices outward from domain center
    center = np.mean(pts, axis=0)
    expanded = []
    for p in hull_pts:
        vec = p - center
        dist = np.linalg.norm(vec)
        unit = vec / (dist + 1e-6)
        expanded.append(p + unit * pad)
        
    expanded = np.array(expanded)
    loop = np.vstack([expanded, expanded[0]])
    
    # Periodic cubic spline produces smooth rounded boundaries with ZERO spikes
    tck, u = splprep([loop[:, 0], loop[:, 1]], s=2.5, per=True)
    u_fine = np.linspace(0, 1, 350)
    sx, sy = splev(u_fine, tck)
    
    ax.fill(sx, sy, color=fill_color, alpha=0.65, zorder=2)
    ax.plot(sx, sy, color=border_color, lw=2.0, ls='--', zorder=2)

# =============================================================================
# 3. MAIN DIAGRAM RENDERER
# =============================================================================
def draw_topology_diagram_v3(topo_key, output_path):
    fig_w, fig_h = 15.0, 10.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    with open(f"data/{topo_key}_topology.json") as f:
        topo_data = json.load(f)

    nodes = topo_data['nodes']
    links = topo_data['links']
    coords = {'nsfnet': NSFNET_COORDS, 'abilene': ABILENE_COORDS, 'geant': GEANT_COORDS}[topo_key]
    curvatures = {'nsfnet': NSFNET_CURVATURES, 'abilene': ABILENE_CURVATURES, 'geant': GEANT_CURVATURES}[topo_key]

    title_map = {
        'geant': ('GÉANT Network (22 Nodes, 72 Directed Links)', 'Source In-Domain Training Topology'),
        'nsfnet': ('NSFNET Network (14 Nodes, 42 Directed Links)', 'Cross-Topology Transfer & Fine-Tuning Target'),
        'abilene': ('Abilene Network (12 Nodes, 30 Directed Links)', 'Cross-Topology Transfer & Zero-Padding Target')
    }
    main_title, sub_title = title_map[topo_key]

    # Title & Subtitle
    ax.text(50, 97.2, main_title, fontsize=17, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(50, 94.6, f"Hierarchical FL-DRL Control Architecture  |  {sub_title}", 
            fontsize=11, ha='center', va='center', color='#64748b')

    # =========================================================================
    # TIER 1: CENTRAL MANAGEMENT PLANE (Cloud / Central NOC)
    # =========================================================================
    top_box = FancyBboxPatch((10, 75.5), 80, 16.5, boxstyle="round,pad=0.8,rounding_size=2.0",
                             ec="#0284c7", fc="#f8fafc", lw=2, zorder=2)
    ax.add_patch(top_box)
    
    ax.text(50, 89.5, "GLOBAL MANAGEMENT PLANE  (Central Cloud / ISP NOC)",
            fontsize=12, fontweight='bold', ha='center', va='center', color="#0369a1")

    # 1. Global FL Server
    fl_srv_box = FancyBboxPatch((13, 77.2), 22, 9.2, boxstyle="round,pad=0.5,rounding_size=1.2",
                                ec="#0284c7", fc="#ffffff", lw=1.5, zorder=3)
    ax.add_patch(fl_srv_box)
    ax.text(24, 83.7, "Global FL Server", fontsize=10.5, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(24, 81.2, "• FedAdam / FedAvg", fontsize=8.5, ha='center', va='center', color='#475569')
    ax.text(24, 79.0, "• Aggregates weights ΔW", fontsize=8, ha='center', va='center', color='#64748b')

    # 2. RecurrentPPO Agent
    rl_box = FancyBboxPatch((38, 77.2), 25, 9.2, boxstyle="round,pad=0.5,rounding_size=1.2",
                            ec="#16a34a", fc="#ffffff", lw=1.5, zorder=3)
    ax.add_patch(rl_box)
    ax.text(50.5, 83.7, "RecurrentPPO TE Agent", fontsize=10.5, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(50.5, 81.2, "• Actor-Critic with LSTM", fontsize=8.5, ha='center', va='center', color='#475569')
    ax.text(50.5, 79.0, "• Computes global TE paths", fontsize=8, ha='center', va='center', color='#64748b')

    # 3. Admission Guard
    guard_box = FancyBboxPatch((66, 77.2), 21, 9.2, boxstyle="round,pad=0.5,rounding_size=1.2",
                               ec="#dc2626", fc="#ffffff", lw=1.5, zorder=3)
    ax.add_patch(guard_box)
    ax.text(76.5, 83.7, "Admission Guard", fontsize=10.5, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(76.5, 81.2, "• Projected Load Verifier", fontsize=8.5, ha='center', va='center', color='#475569')
    ax.text(76.5, 79.0, "• Fallback to Shortest Path", fontsize=8, ha='center', va='center', color='#64748b')

    # Side badges for Transfer & Fine-Tuning
    if topo_key == 'nsfnet':
        t_box = FancyBboxPatch((1.5, 76), 7.5, 15.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                               ec="#9333ea", fc="#faf5ff", lw=1.5, zorder=3)
        ax.add_patch(t_box)
        ax.text(5.25, 88.0, "GÉANT", fontsize=9, fontweight='bold', ha='center', va='center', color='#7e22ce')
        ax.text(5.25, 85.5, "Weights", fontsize=8.5, ha='center', va='center', color='#7e22ce')
        ax.text(5.25, 81.0, "Zero-Shot\nTransfer\n&\nFine-Tune", fontsize=7.5, ha='center', va='center', color='#6b21a8')
        arrow = FancyArrowPatch((9.2, 83.5), (12.8, 83.5), arrowstyle='->', mutation_scale=15, color='#9333ea', lw=2)
        ax.add_patch(arrow)

    elif topo_key == 'abilene':
        t_box = FancyBboxPatch((1.5, 76), 7.5, 15.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                               ec="#9333ea", fc="#faf5ff", lw=1.5, zorder=3)
        ax.add_patch(t_box)
        ax.text(5.25, 88.0, "GÉANT", fontsize=9, fontweight='bold', ha='center', va='center', color='#7e22ce')
        ax.text(5.25, 85.5, "Weights", fontsize=8.5, ha='center', va='center', color='#7e22ce')
        ax.text(5.25, 80.5, "Zero-Padding\n(174d → 75d)\n&\nFine-Tune", fontsize=7, ha='center', va='center', color='#6b21a8')
        arrow = FancyArrowPatch((9.2, 83.5), (12.8, 83.5), arrowstyle='->', mutation_scale=15, color='#9333ea', lw=2)
        ax.add_patch(arrow)

    # =========================================================================
    # TIER 2: REGIONAL CONTROL LAYER (Domain Controllers / FL Clients)
    # =========================================================================
    ctrl_specs = [
        ('A', 22, COLOR_A, BG_A, "Domain A SDN Controller\n(FL Client A: TrafficLSTM)"),
        ('B', 50, COLOR_B, BG_B, "Domain B SDN Controller\n(FL Client B: TrafficLSTM)"),
        ('C', 78, COLOR_C, BG_C, "Domain C SDN Controller\n(FL Client C: TrafficLSTM)")
    ]

    ctrl_centers = {}
    for dom, cx, col, bg, label in ctrl_specs:
        c_box = FancyBboxPatch((cx - 12, 60.5), 24, 7.8, boxstyle="round,pad=0.5,rounding_size=1.4",
                               ec=col, fc=bg, lw=2, zorder=3)
        ax.add_patch(c_box)
        ax.text(cx, 64.4, label, fontsize=9.5, fontweight='bold', ha='center', va='center', color='#0f172a')
        ctrl_centers[dom] = (cx, 60.5)

        # FL Communication arrows with Central Server
        ax.annotate('', xy=(cx - 2, 75.3), xytext=(cx - 2, 68.3),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.6, ls="--"))
        ax.annotate('', xy=(cx + 2, 68.3), xytext=(cx + 2, 75.3),
                    arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.6))

    # Communication legend
    ax.text(91.5, 71.5, "─── Global Model (W)", fontsize=8, color="#0284c7", ha='left', va='center', fontweight='bold')
    ax.text(91.5, 69.2, "- - - Local Weights (ΔW)", fontsize=8, color="#475569", ha='left', va='center', fontweight='bold')
    ax.text(91.5, 67.0, "━━━ OpenFlow 1.3", fontsize=8, color="#1e293b", ha='left', va='center', fontweight='bold')

    # =========================================================================
    # TIER 3: DATA PLANE (Network Topology with 100% UNIFIED DOMAIN ZONES)
    # =========================================================================
    dp_bg = FancyBboxPatch((5, 5.0), 90, 48.5, boxstyle="round,pad=0.8,rounding_size=2.0",
                           ec="#94a3b8", fc="#ffffff", lw=1.5, zorder=1)
    ax.add_patch(dp_bg)
    
    ax.text(7.5, 51.5, f"DATA PLANE: {topo_key.upper()} TOPOLOGY (Switches & Physical Links)",
            fontsize=11.5, fontweight='bold', ha='left', va='center', color="#1e293b")
    ax.text(7.5, 49.3, "Partitioned into 3 Autonomous Administrative Domains (Domain A, B, C)",
            fontsize=9, ha='left', va='center', color="#64748b")

    # Draw 100% UNIFIED, continuous domain shapes (never split!)
    domain_nodes_dict = {
        dom: [n['id'] for n in nodes if n['domain'] == dom]
        for dom in ['A', 'B', 'C']
    }

    for dom in ['A', 'B', 'C']:
        dom_pts = [coords[nid] for nid in domain_nodes_dict[dom]]
        draw_unified_domain_zone(ax, dom_pts, DOMAIN_COLORS[dom], DOMAIN_BG[dom], pad=4.8)

    # Southbound OpenFlow Control: clean connections to the top of each domain zone
    for dom in ['A', 'B', 'C']:
        cx, cy = ctrl_centers[dom]
        dom_pts = [coords[nid] for nid in domain_nodes_dict[dom]]
        avg_x = np.mean([p[0] for p in dom_pts])
        # Find highest point of that domain to connect cleanly without cutting across links
        max_y = max(p[1] for p in dom_pts) + 4.8
        
        ax.annotate('', xy=(avg_x, max_y), xytext=(cx, 60.5),
                    arrowprops=dict(arrowstyle="<->", color=DOMAIN_COLORS[dom], lw=2.2, alpha=0.9))
        
        # Pill badge on arrow
        mid_x, mid_y = (cx + avg_x) / 2, (60.5 + max_y) / 2
        badge = FancyBboxPatch((mid_x - 5.5, mid_y - 1.2), 11, 2.4, boxstyle="round,pad=0.2,rounding_size=0.6",
                               ec=DOMAIN_COLORS[dom], fc="#ffffff", lw=1.2, zorder=4)
        ax.add_patch(badge)
        ax.text(mid_x, mid_y, "OpenFlow 1.3", fontsize=7.5, ha='center', va='center',
                color=DOMAIN_COLORS[dom], fontweight='bold', zorder=5)

    # Draw Links (Curved for crossing links, straight for direct adjacent links)
    for link in links:
        u, v = link['source'], link['target']
        p1, p2 = coords[u], coords[v]
        dom_u = nodes[u]['domain']
        dom_v = nodes[v]['domain']
        
        if dom_u == dom_v:
            edge_col = DOMAIN_COLORS[dom_u]
            edge_lw = 2.4
            edge_alpha = 0.95
        else:
            edge_col = "#64748b"
            edge_lw = 2.0
            edge_alpha = 0.85
            
        rad = curvatures.get((u, v), curvatures.get((v, u), 0.0))
        if rad != 0.0:
            arc = FancyArrowPatch(posA=p1, posB=p2, connectionstyle=f"arc3,rad={rad}",
                                  arrowstyle='-', color=edge_col, lw=edge_lw, alpha=edge_alpha, zorder=3)
            ax.add_patch(arc)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge_col, lw=edge_lw, alpha=edge_alpha, zorder=3)

    # Draw Nodes: 100% Mathematically Perfect Circles
    node_xs = [coords[n['id']][0] for n in nodes]
    node_ys = [coords[n['id']][1] for n in nodes]
    node_colors = [DOMAIN_COLORS[n['domain']] for n in nodes]

    # Halo + Disc
    ax.scatter(node_xs, node_ys, s=850, c='#ffffff', edgecolors=node_colors, linewidths=2.8, zorder=5)
    ax.scatter(node_xs, node_ys, s=550, c=node_colors, zorder=6)

    # Node Labels
    for n in nodes:
        nid = n['id']
        px, py = coords[nid]
        lbl = n['label']
        ax.text(px, py, f"S{nid}", fontsize=8.5, fontweight='bold', ha='center', va='center', color='#ffffff', zorder=7)
        ax.text(px, py - 3.2, lbl, fontsize=8, fontweight='bold', ha='center', va='top', color='#0f172a', zorder=7)

    # =========================================================================
    # FOOTER BANNER (IRL vs Demo Distinction)
    # =========================================================================
    footer_box = FancyBboxPatch((5, 1.0), 90, 3.2, boxstyle="round,pad=0.2,rounding_size=0.8",
                                ec="#cbd5e1", fc="#f8fafc", lw=1.2, zorder=2)
    ax.add_patch(footer_box)
    
    info_text = (
        r"$\bf{REAL\ WORLD\ (Production):}$ Edge SDN Controllers run in each region; FL Server & RecurrentPPO run in central Cloud/NOC."
        "\n"
        r"$\bf{DEMO\ (Prototype):}$ High-speed analytical simulation (Gymnasium + NumPy + Yen's KSP) executed in-memory on 1 PC."
    )
    ax.text(50, 2.6, info_text, fontsize=8.5, ha='center', va='center', color='#1e293b', linespacing=1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated clean v3 diagram: {output_path}")

if __name__ == '__main__':
    for topo in ['geant', 'nsfnet', 'abilene']:
        draw_topology_diagram_v3(topo, f"report/figures/topology_deployment_{topo}.png")
