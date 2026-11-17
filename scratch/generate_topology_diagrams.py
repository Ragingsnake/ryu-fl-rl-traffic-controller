import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import networkx as nx
import numpy as np

# Set font and aesthetics
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# Colors for domains
COLOR_A = "#1f77b4"  # Blue - Domain A
COLOR_B = "#ff7f0e"  # Orange - Domain B
COLOR_C = "#2ca02c"  # Green - Domain C

BG_A = "#e6f2fa"
BG_B = "#fff3e6"
BG_C = "#eaf7ea"

DOMAIN_COLORS = {'A': COLOR_A, 'B': COLOR_B, 'C': COLOR_C}
DOMAIN_BG = {'A': BG_A, 'B': BG_B, 'C': BG_C}

# Approximate geographic coordinates (longitude, latitude normalized for layout)
NSFNET_COORDS = {
    0: (-122.3, 47.6),   # WA (Seattle)
    1: (-122.1, 37.4),   # CA1 (Palo Alto)
    2: (-117.2, 32.7),   # CA2 (San Diego)
    3: (-96.7, 40.8),    # NE (Lincoln)
    4: (-88.2, 40.1),    # IL (Champaign)
    5: (-83.7, 42.3),    # MI (Ann Arbor)
    6: (-84.4, 33.7),    # GA (Atlanta)
    7: (-95.4, 29.8),    # TX (Houston)
    8: (-111.9, 40.8),   # UT (Salt Lake City)
    9: (-105.3, 40.0),   # CO (Boulder)
    10: (-80.0, 40.4),   # PA (Pittsburgh)
    11: (-76.5, 42.4),   # NY (Ithaca)
    12: (-74.7, 40.3),   # NJ (Princeton)
    13: (-77.0, 38.9),   # DC (College Park)
}

ABILENE_COORDS = {
    10: (-122.3, 47.6),  # STTL (Seattle)
    9: (-122.0, 37.4),   # SNVA (Sunnyvale)
    7: (-118.2, 34.1),   # LOSA (Los Angeles)
    3: (-105.0, 39.7),   # DNVR (Denver)
    6: (-94.6, 39.1),    # KSCY (Kansas City)
    4: (-95.4, 29.8),    # HSTN (Houston)
    2: (-87.6, 41.9),    # CHIN (Chicago)
    5: (-86.2, 39.8),    # IPLS (Indianapolis)
    0: (-84.0, 32.8),    # ATLAM5 (Atlanta M5)
    1: (-84.4, 34.5),    # ATLA (Atlanta)
    11: (-77.0, 38.9),   # WASH (Washington DC)
    8: (-74.0, 40.7),    # NYCM (New York City)
}

GEANT_COORDS = {
    11: (-6.3, 53.3),    # IE (Dublin)
    21: (-0.1, 51.5),    # UK (London)
    18: (-9.1, 38.7),    # PT (Lisbon)
    6: (-3.7, 40.4),     # ES (Madrid)
    7: (2.3, 48.9),      # FR (Paris)
    1: (4.4, 50.8),      # BE (Brussels)
    15: (4.9, 52.4),     # NL (Amsterdam)
    14: (6.1, 49.6),     # LU (Luxembourg)
    2: (6.1, 46.2),      # CH (Geneva)
    4: (8.7, 50.1),      # DE (Frankfurt)
    13: (9.2, 45.5),     # IT (Milan)
    16: (10.8, 59.9),    # NO (Oslo)
    5: (12.6, 55.7),     # DK (Copenhagen)
    0: (16.4, 48.2),     # AT (Vienna)
    3: (14.4, 50.1),     # CZ (Prague)
    20: (14.5, 46.1),    # SI (Ljubljana)
    9: (16.0, 45.8),     # HR (Zagreb)
    10: (19.0, 47.5),    # HU (Budapest)
    17: (21.0, 52.2),    # PL (Warsaw)
    19: (18.1, 59.3),    # SE (Stockholm)
    8: (23.7, 38.0),     # GR (Athens)
    12: (26.0, 34.0),    # IL (Tel Aviv - adjusted slightly for European canvas balance)
}

def normalize_coords(raw_coords, target_box):
    """Normalize raw coords into [xmin, xmax, ymin, ymax]"""
    xmin, xmax, ymin, ymax = target_box
    xs = [pt[0] for pt in raw_coords.values()]
    ys = [pt[1] for pt in raw_coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    norm = {}
    for nid, (x, y) in raw_coords.items():
        nx = xmin + (x - min_x) / (max_x - min_x + 1e-6) * (xmax - xmin)
        ny = ymin + (y - min_y) / (max_y - min_y + 1e-6) * (ymax - ymin)
        norm[nid] = (nx, ny)
    return norm

def draw_topology_deployment(topo_key, output_path):
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Load topology data
    with open(f"data/{topo_key}_topology.json") as f:
        topo_data = json.load(f)

    nodes = topo_data['nodes']
    links = topo_data['links']
    n_nodes = len(nodes)
    n_links = len(links) * 2  # directed links

    title_map = {
        'geant': ('GÉANT Network (22 Nodes, 72 Directed Links)', 'Source In-Domain Training Topology'),
        'nsfnet': ('NSFNET Network (14 Nodes, 42 Directed Links)', 'Cross-Topology Transfer & Fine-Tuning Target'),
        'abilene': ('Abilene Network (12 Nodes, 30 Directed Links)', 'Cross-Topology Transfer & Zero-Padding Target')
    }
    main_title, sub_title = title_map[topo_key]

    # Header Title
    ax.text(50, 96.5, main_title, fontsize=17, fontweight='bold', ha='center', va='center', color='#1e293b')
    ax.text(50, 93.8, f"Hierarchical FL-DRL Control & Deployment Architecture  |  {sub_title}", 
            fontsize=11, ha='center', va='center', color='#64748b', style='italic')

    # =========================================================================
    # TIER 1: CENTRAL MANAGEMENT & ORCHESTRATION PLANE (Cloud / Central NOC)
    # =========================================================================
    top_box = FancyBboxPatch((12, 73), 76, 17, boxstyle="round,pad=0.8,rounding_size=2.0",
                             ec="#0284c7", fc="#f0f9ff", lw=2, zorder=2)
    ax.add_patch(top_box)
    
    # Plane Header
    ax.text(50, 87.5, "CENTRAL MANAGEMENT PLANE  (Global Orchestrator / Central Cloud)",
            fontsize=12, fontweight='bold', ha='center', va='center', color="#0369a1")

    # Inner Components of Central Management Plane
    # 1. FL Server
    fl_srv_box = FancyBboxPatch((15, 75), 20, 9.5, boxstyle="round,pad=0.4,rounding_size=1.2",
                                ec="#0284c7", fc="#ffffff", lw=1.5, zorder=3)
    ax.add_patch(fl_srv_box)
    ax.text(25, 82, "Global FL Server", fontsize=10, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(25, 79.5, "• FedAdam / FedAvg", fontsize=8.5, ha='center', va='center', color='#475569')
    ax.text(25, 77.2, "• Aggregates weights ΔW", fontsize=8, ha='center', va='center', color='#64748b')

    # 2. RL Agent (RecurrentPPO)
    rl_box = FancyBboxPatch((38, 75), 24, 9.5, boxstyle="round,pad=0.4,rounding_size=1.2",
                            ec="#16a34a", fc="#ffffff", lw=1.5, zorder=3)
    ax.add_patch(rl_box)
    ax.text(50, 82, "RecurrentPPO TE Agent", fontsize=10, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(50, 79.5, "• LSTM Memory (History)", fontsize=8.5, ha='center', va='center', color='#475569')
    ax.text(50, 77.2, "• Computes global TE paths", fontsize=8, ha='center', va='center', color='#64748b')

    # 3. Admission Guard
    guard_box = FancyBboxPatch((65, 75), 20, 9.5, boxstyle="round,pad=0.4,rounding_size=1.2",
                               ec="#dc2626", fc="#ffffff", lw=1.5, zorder=3)
    ax.add_patch(guard_box)
    ax.text(75, 82, "Admission Guard", fontsize=10, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(75, 79.5, "• Safety Load Verifier", fontsize=8.5, ha='center', va='center', color='#475569')
    ax.text(75, 77.2, "• Fallback to Shortest Path", fontsize=8, ha='center', va='center', color='#64748b')

    # Optional Transfer learning indicator for NSFNET and Abilene
    if topo_key == 'nsfnet':
        trans_box = FancyBboxPatch((1.5, 75.5), 8.5, 12, boxstyle="round,pad=0.4,rounding_size=1.0",
                                  ec="#9333ea", fc="#faf5ff", lw=1.5, zorder=3)
        ax.add_patch(trans_box)
        ax.text(5.75, 84, "GÉANT", fontsize=9, fontweight='bold', ha='center', va='center', color='#7e22ce')
        ax.text(5.75, 81.5, "Weights", fontsize=8, ha='center', va='center', color='#7e22ce')
        ax.text(5.75, 78, "Zero-Shot\n& Fine-Tune", fontsize=7.5, ha='center', va='center', color='#6b21a8')
        arrow = FancyArrowPatch((10, 81.5), (14.5, 81.5), arrowstyle='->', mutation_scale=15, color='#9333ea', lw=2)
        ax.add_patch(arrow)

    elif topo_key == 'abilene':
        trans_box = FancyBboxPatch((1.5, 75.5), 8.5, 12, boxstyle="round,pad=0.4,rounding_size=1.0",
                                  ec="#9333ea", fc="#faf5ff", lw=1.5, zorder=3)
        ax.add_patch(trans_box)
        ax.text(5.75, 84, "GÉANT", fontsize=9, fontweight='bold', ha='center', va='center', color='#7e22ce')
        ax.text(5.75, 81.5, "Weights", fontsize=8, ha='center', va='center', color='#7e22ce')
        ax.text(5.75, 78, "Zero-Padding\n(174d → 75d)", fontsize=7, ha='center', va='center', color='#6b21a8')
        arrow = FancyArrowPatch((10, 81.5), (14.5, 81.5), arrowstyle='->', mutation_scale=15, color='#9333ea', lw=2)
        ax.add_patch(arrow)

    # =========================================================================
    # TIER 2: REGIONAL CONTROL LAYER (Domain Controllers / FL Clients)
    # =========================================================================
    ctrl_specs = [
        ('A', 20, COLOR_A, BG_A, "Domain A SDN Controller\n(FL Client A: TrafficLSTM)"),
        ('B', 50, COLOR_B, BG_B, "Domain B SDN Controller\n(FL Client B: TrafficLSTM)"),
        ('C', 80, COLOR_C, BG_C, "Domain C SDN Controller\n(FL Client C: TrafficLSTM)")
    ]

    ctrl_centers = {}
    for dom, cx, col, bg, label in ctrl_specs:
        c_box = FancyBboxPatch((cx - 12, 57.5), 24, 8, boxstyle="round,pad=0.5,rounding_size=1.5",
                               ec=col, fc=bg, lw=1.8, zorder=3)
        ax.add_patch(c_box)
        ax.text(cx, 61.5, label, fontsize=9, fontweight='bold', ha='center', va='center', color='#0f172a')
        ctrl_centers[dom] = (cx, 57.5)

        # Arrows between Central Plane and Domain Controllers
        # Weight upload (ΔW) and Model download (W_global)
        ax.annotate('', xy=(cx - 2, 72.8), xytext=(cx - 2, 65.7),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4, ls="--"))
        ax.annotate('', xy=(cx + 2, 65.7), xytext=(cx + 2, 72.8),
                    arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.4))

    # Communication legend near Tier 2
    ax.text(92.5, 69.5, "─── Global Model (W)", fontsize=7.5, color="#0284c7", ha='left', va='center')
    ax.text(92.5, 67.5, "- - - Local Weights (ΔW)", fontsize=7.5, color="#475569", ha='left', va='center')
    ax.text(92.5, 65.5, "━━━ OpenFlow 1.3", fontsize=7.5, color="#334155", ha='left', va='center')

    # =========================================================================
    # TIER 3: DATA PLANE (Network Topology with Domain Partitioning)
    # =========================================================================
    # Background card for data plane
    dp_bg = FancyBboxPatch((5, 4), 90, 48, boxstyle="round,pad=0.8,rounding_size=2.0",
                           ec="#cbd5e1", fc="#ffffff", lw=1.5, zorder=1)
    ax.add_patch(dp_bg)
    ax.text(8.5, 49.5, f"DATA PLANE: {topo_key.upper()} TOPOLOGY (Routers & Physical Links)",
            fontsize=11, fontweight='bold', ha='left', va='center', color="#334155")
    ax.text(8.5, 47.3, "Partitioned into 3 Autonomous Administrative Domains (A, B, C)",
            fontsize=8.5, ha='left', va='center', color="#64748b")

    # Compute node positions
    raw_coords = {'nsfnet': NSFNET_COORDS, 'abilene': ABILENE_COORDS, 'geant': GEANT_COORDS}[topo_key]
    # Target box in canvas: x: 10 to 90, y: 8 to 44
    coords = normalize_coords(raw_coords, (10, 88, 8, 43))

    # Draw shaded domain hulls/areas
    for dom, col, bg in [('A', COLOR_A, BG_A), ('B', COLOR_B, BG_B), ('C', COLOR_C, BG_C)]:
        dom_node_pts = [coords[n['id']] for n in nodes if n['domain'] == dom]
        if len(dom_node_pts) >= 3:
            # Expand points slightly for visual margin
            center = np.mean(dom_node_pts, axis=0)
            hull_pts = []
            # Sort by angle around center
            angles = [np.arctan2(p[1] - center[1], p[0] - center[0]) for p in dom_node_pts]
            sorted_idx = np.argsort(angles)
            for idx in sorted_idx:
                p = dom_node_pts[idx]
                vec = np.array(p) - center
                expanded = center + vec * 1.35
                hull_pts.append(expanded)
            poly = Polygon(hull_pts, closed=True, fc=bg, ec=col, lw=1.2, ls=":", alpha=0.55, zorder=2)
            ax.add_patch(poly)

    # Draw Links
    for link in links:
        u, v = link['source'], link['target']
        pu, pv = coords[u], coords[v]
        # Link color: if both nodes same domain, use domain color, else gray inter-domain
        dom_u = nodes[u]['domain']
        dom_v = nodes[v]['domain']
        edge_col = DOMAIN_COLORS[dom_u] if dom_u == dom_v else "#94a3b8"
        edge_lw = 2.0 if dom_u == dom_v else 1.6
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]], color=edge_col, lw=edge_lw, zorder=3, alpha=0.85)

    # Southbound Control links (Domain Controller to representative domain nodes)
    # Draw dashed lines from each domain controller to its nodes
    for dom in ['A', 'B', 'C']:
        cx, cy = ctrl_centers[dom]
        dom_node_ids = [n['id'] for n in nodes if n['domain'] == dom]
        # Connect controller to central node of that domain
        dom_pts = [coords[nid] for nid in dom_node_ids]
        avg_pt = np.mean(dom_pts, axis=0)
        ax.annotate('', xy=(avg_pt[0], avg_pt[1] + 3), xytext=(cx, 57.5),
                    arrowprops=dict(arrowstyle="<->", color=DOMAIN_COLORS[dom], lw=1.8, ls="-", alpha=0.9))
        ax.text((cx + avg_pt[0]) / 2 + 1.5, (57.5 + avg_pt[1]) / 2, "OpenFlow\nControl",
                fontsize=7, ha='left', va='center', color=DOMAIN_COLORS[dom], fontweight='bold')

    # Draw Nodes
    for n in nodes:
        nid = n['id']
        lbl = n['label']
        dom = n['domain']
        px, py = coords[nid]
        col = DOMAIN_COLORS[dom]

        # Outer ring
        circle_bg = plt.Circle((px, py), 1.9, fc="#ffffff", ec=col, lw=2.2, zorder=5)
        ax.add_patch(circle_bg)
        inner_circle = plt.Circle((px, py), 1.4, fc=col, zorder=6)
        ax.add_patch(inner_circle)

        # Label inside node or next to it
        ax.text(px, py, f"S{nid}", fontsize=7.5, fontweight='bold', ha='center', va='center', color='#ffffff', zorder=7)
        # City/Label text
        ax.text(px, py - 2.8, lbl, fontsize=7.5, fontweight='semibold', ha='center', va='top', color='#1e293b', zorder=7)

    # =========================================================================
    # BOTTOM SUMMARY BANNER (IRL vs Demo Distinction)
    # =========================================================================
    summary_box = FancyBboxPatch((5, 0.8), 90, 2.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                                 ec="#e2e8f0", fc="#f8fafc", lw=1, zorder=2)
    ax.add_patch(summary_box)
    
    info_text = (
        r"$\bf{IN\ REAL\ LIFE:}$ Regional SDN Controllers run at each edge region; FL Server & RL TE Orchestrator run in central NOC/Cloud. "
        r"$\bf{IN\ DEMO:}$ All components run in-memory inside Python on 1 PC."
    )
    ax.text(50, 2.05, info_text, fontsize=8, ha='center', va='center', color='#334155')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

if __name__ == '__main__':
    for topo in ['geant', 'nsfnet', 'abilene']:
        draw_topology_deployment(topo, f"report/figures/topology_deployment_{topo}.png")
