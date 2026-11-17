import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set font styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# Colors for domains (Modern, accessible palette)
COLOR_A = "#0284c7"  # Deep Sky Blue
COLOR_B = "#d97706"  # Warm Amber
COLOR_C = "#059669"  # Emerald Green

BG_A = "#e0f2fe"
BG_B = "#fef3c7"
BG_C = "#d1fae5"

DOMAIN_COLORS = {'A': COLOR_A, 'B': COLOR_B, 'C': COLOR_C}
DOMAIN_BG = {'A': BG_A, 'B': BG_B, 'C': BG_C}

# Carefully balanced node coordinates for Data Plane (scaled nicely to fill x: 8..92, y: 10..48)
NSFNET_COORDS = {
    0: (14, 45),   # WA
    1: (12, 31),   # CA1
    2: (15, 17),   # CA2
    8: (25, 33),   # UT
    7: (42, 13),   # TX
    9: (36, 32),   # CO
    3: (46, 35),   # NE
    4: (56, 33),   # IL
    5: (64, 43),   # MI
    6: (71, 15),   # GA
    10: (75, 34),  # PA
    11: (83, 44),  # NY
    12: (88, 36),  # NJ
    13: (83, 26),  # DC
}

ABILENE_COORDS = {
    10: (15, 45),  # STTL
    9: (13, 31),   # SNVA
    7: (17, 18),   # LOSA
    3: (32, 33),   # DNVR
    6: (47, 32),   # KSCY
    2: (57, 43),   # CHIN
    5: (63, 33),   # IPLS
    4: (48, 14),   # HSTN
    0: (71, 14),   # ATLAM5
    1: (74, 23),   # ATLA
    11: (82, 30),  # WASH
    8: (88, 42),   # NYCM
}

GEANT_COORDS = {
    11: (14, 38),  # IE
    21: (23, 37),  # UK
    7: (26, 26),   # FR
    1: (31, 35),   # BE
    15: (35, 43),  # NL
    16: (38, 48),  # NO
    19: (48, 48),  # SE
    4: (41, 33),   # DE
    2: (35, 23),   # CH
    3: (51, 34),   # CZ
    0: (54, 25),   # AT
    17: (63, 42),  # PL
    10: (64, 26),  # HU
    20: (50, 16),  # SI
    9: (58, 15),   # HR
    18: (11, 13),  # PT
    6: (19, 15),   # ES
    14: (34, 30),  # LU
    13: (43, 19),  # IT
    5: (44, 43),   # DK
    8: (75, 14),   # GR
    12: (87, 12),  # IL
}

def draw_topology_diagram(topo_key, output_path):
    fig_w, fig_h = 15.0, 10.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Load data
    with open(f"data/{topo_key}_topology.json") as f:
        topo_data = json.load(f)

    nodes = topo_data['nodes']
    links = topo_data['links']
    coords = {'nsfnet': NSFNET_COORDS, 'abilene': ABILENE_COORDS, 'geant': GEANT_COORDS}[topo_key]

    title_map = {
        'geant': ('GÉANT Network (22 Nodes, 72 Directed Links)', 'Source In-Domain Training Topology'),
        'nsfnet': ('NSFNET Network (14 Nodes, 42 Directed Links)', 'Cross-Topology Transfer & Fine-Tuning Target'),
        'abilene': ('Abilene Network (12 Nodes, 30 Directed Links)', 'Cross-Topology Transfer & Zero-Padding Target')
    }
    main_title, sub_title = title_map[topo_key]

    # Header Title
    ax.text(50, 97.2, main_title, fontsize=17, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(50, 94.6, f"Hierarchical FL-DRL Control Architecture  |  {sub_title}", 
            fontsize=11, ha='center', va='center', color='#64748b')

    # =========================================================================
    # TIER 1: CENTRAL MANAGEMENT PLANE (Cloud / Central NOC)
    # =========================================================================
    top_box = FancyBboxPatch((10, 75), 80, 16.5, boxstyle="round,pad=0.8,rounding_size=2.0",
                             ec="#0284c7", fc="#f8fafc", lw=2, zorder=2)
    ax.add_patch(top_box)
    
    ax.text(50, 89.2, "GLOBAL MANAGEMENT PLANE  (Central Cloud / ISP NOC)",
            fontsize=12, fontweight='bold', ha='center', va='center', color="#0369a1")

    # Module 1: FL Server
    fl_srv_box = FancyBboxPatch((13, 77), 22, 9.2, boxstyle="round,pad=0.5,rounding_size=1.2",
                                ec="#0284c7", fc="#ffffff", lw=1.5, zorder=3)
    ax.add_patch(fl_srv_box)
    ax.text(24, 83.5, "Global FL Server", fontsize=10.5, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(24, 81.0, "• FedAdam / FedAvg", fontsize=8.5, ha='center', va='center', color='#475569')
    ax.text(24, 78.8, "• Aggregates weights ΔW", fontsize=8, ha='center', va='center', color='#64748b')

    # Module 2: RecurrentPPO Agent
    rl_box = FancyBboxPatch((38, 77), 25, 9.2, boxstyle="round,pad=0.5,rounding_size=1.2",
                            ec="#16a34a", fc="#ffffff", lw=1.5, zorder=3)
    ax.add_patch(rl_box)
    ax.text(50.5, 83.5, "RecurrentPPO TE Agent", fontsize=10.5, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(50.5, 81.0, "• Actor-Critic with LSTM", fontsize=8.5, ha='center', va='center', color='#475569')
    ax.text(50.5, 78.8, "• Computes global TE paths", fontsize=8, ha='center', va='center', color='#64748b')

    # Module 3: Admission Guard
    guard_box = FancyBboxPatch((66, 77), 21, 9.2, boxstyle="round,pad=0.5,rounding_size=1.2",
                               ec="#dc2626", fc="#ffffff", lw=1.5, zorder=3)
    ax.add_patch(guard_box)
    ax.text(76.5, 83.5, "Admission Guard", fontsize=10.5, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(76.5, 81.0, "• Projected Load Verifier", fontsize=8.5, ha='center', va='center', color='#475569')
    ax.text(76.5, 78.8, "• Fallback to Shortest Path", fontsize=8, ha='center', va='center', color='#64748b')

    # Side badges for Transfer & Fine-Tuning
    if topo_key == 'nsfnet':
        t_box = FancyBboxPatch((1.5, 76), 7.5, 14.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                               ec="#9333ea", fc="#faf5ff", lw=1.5, zorder=3)
        ax.add_patch(t_box)
        ax.text(5.25, 87.5, "GÉANT", fontsize=9, fontweight='bold', ha='center', va='center', color='#7e22ce')
        ax.text(5.25, 85.0, "Weights", fontsize=8.5, ha='center', va='center', color='#7e22ce')
        ax.text(5.25, 81.0, "Zero-Shot\nTransfer\n&\nFine-Tune", fontsize=7.5, ha='center', va='center', color='#6b21a8')
        arrow = FancyArrowPatch((9.2, 83), (12.8, 83), arrowstyle='->', mutation_scale=15, color='#9333ea', lw=2)
        ax.add_patch(arrow)

    elif topo_key == 'abilene':
        t_box = FancyBboxPatch((1.5, 76), 7.5, 14.5, boxstyle="round,pad=0.4,rounding_size=1.0",
                               ec="#9333ea", fc="#faf5ff", lw=1.5, zorder=3)
        ax.add_patch(t_box)
        ax.text(5.25, 87.5, "GÉANT", fontsize=9, fontweight='bold', ha='center', va='center', color='#7e22ce')
        ax.text(5.25, 85.0, "Weights", fontsize=8.5, ha='center', va='center', color='#7e22ce')
        ax.text(5.25, 80.5, "Zero-Padding\n(174d → 75d)\n&\nFine-Tune", fontsize=7, ha='center', va='center', color='#6b21a8')
        arrow = FancyArrowPatch((9.2, 83), (12.8, 83), arrowstyle='->', mutation_scale=15, color='#9333ea', lw=2)
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
        c_box = FancyBboxPatch((cx - 12, 60), 24, 7.8, boxstyle="round,pad=0.5,rounding_size=1.4",
                               ec=col, fc=bg, lw=2, zorder=3)
        ax.add_patch(c_box)
        ax.text(cx, 63.9, label, fontsize=9.5, fontweight='bold', ha='center', va='center', color='#0f172a')
        ctrl_centers[dom] = (cx, 60)

        # Arrows between Central Plane and Regional Controllers
        # Local model weights upload (ΔW)
        ax.annotate('', xy=(cx - 2, 74.8), xytext=(cx - 2, 67.8),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.6, ls="--"))
        # Global model distribution (W_global)
        ax.annotate('', xy=(cx + 2, 67.8), xytext=(cx + 2, 74.8),
                    arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.6))

    # Communication legend
    ax.text(91.5, 71.5, "─── Global Model (W)", fontsize=8, color="#0284c7", ha='left', va='center', fontweight='bold')
    ax.text(91.5, 69.2, "- - - Local Weights (ΔW)", fontsize=8, color="#475569", ha='left', va='center', fontweight='bold')
    ax.text(91.5, 67.0, "━━━ OpenFlow 1.3", fontsize=8, color="#1e293b", ha='left', va='center', fontweight='bold')

    # =========================================================================
    # TIER 3: DATA PLANE (Network Topology with Smooth Gaussian Domain Zones)
    # =========================================================================
    dp_bg = FancyBboxPatch((5, 5.5), 90, 48, boxstyle="round,pad=0.8,rounding_size=2.0",
                           ec="#94a3b8", fc="#ffffff", lw=1.5, zorder=1)
    ax.add_patch(dp_bg)
    
    ax.text(7.5, 51.5, f"DATA PLANE: {topo_key.upper()} TOPOLOGY (Switches & Links)",
            fontsize=11.5, fontweight='bold', ha='left', va='center', color="#1e293b")
    ax.text(7.5, 49.3, "Partitioned into 3 Autonomous Administrative Domains (Domain A, B, C)",
            fontsize=9, ha='left', va='center', color="#64748b")

    # Smooth Gaussian Metaball Contours for Domains
    # Meshgrid covering the data plane area
    gx, gy = np.meshgrid(np.linspace(6, 94, 400), np.linspace(6, 50, 300))
    # Correct for canvas aspect ratio so circles and blobs are perfectly isotropic in screen space
    aspect_factor = fig_w / fig_h  # 15 / 10.5 = 1.42857

    domain_nodes_dict = {
        dom: [n['id'] for n in nodes if n['domain'] == dom]
        for dom in ['A', 'B', 'C']
    }

    sigma = 5.2
    thresh = 0.38

    for dom in ['A', 'B', 'C']:
        nids = domain_nodes_dict[dom]
        col_border = DOMAIN_COLORS[dom]
        col_fill = DOMAIN_BG[dom]

        z = np.zeros_like(gx)
        for nid in nids:
            px, py = coords[nid]
            # Isotropic Euclidean distance squared in screen coordinates
            dist_sq = (gx - px)**2 + ((gy - py) * aspect_factor)**2
            z += np.exp(-dist_sq / (2 * sigma**2))

        # Smooth filled contour and smooth dashed border
        ax.contourf(gx, gy, z, levels=[thresh, 999], colors=[col_fill], alpha=0.7, zorder=2)
        ax.contour(gx, gy, z, levels=[thresh], colors=[col_border], linewidths=2.0, linestyles='--', zorder=2)

    # Southbound OpenFlow Control arrows from controllers to domain centroids
    for dom in ['A', 'B', 'C']:
        cx, cy = ctrl_centers[dom]
        dom_pts = [coords[nid] for nid in domain_nodes_dict[dom]]
        avg_x = np.mean([p[0] for p in dom_pts])
        avg_y = np.mean([p[1] for p in dom_pts])
        
        # Draw clean arrow from controller base to domain centroid
        ax.annotate('', xy=(avg_x, avg_y + 4.5), xytext=(cx, 60),
                    arrowprops=dict(arrowstyle="<->", color=DOMAIN_COLORS[dom], lw=2.2, alpha=0.9))
        ax.text((cx + avg_x) / 2 + 1.2, (60 + avg_y) / 2 + 2, "OpenFlow\nControl",
                fontsize=7.5, ha='left', va='center', color=DOMAIN_COLORS[dom], fontweight='bold')

    # Draw Links
    for link in links:
        u, v = link['source'], link['target']
        p1, p2 = coords[u], coords[v]
        dom_u = nodes[u]['domain']
        dom_v = nodes[v]['domain']
        
        if dom_u == dom_v:
            # Intra-domain link
            edge_col = DOMAIN_COLORS[dom_u]
            edge_lw = 2.6
            edge_alpha = 0.95
        else:
            # Inter-domain link
            edge_col = "#64748b"
            edge_lw = 2.0
            edge_alpha = 0.8
            
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge_col, lw=edge_lw, alpha=edge_alpha, zorder=3)

    # Draw Nodes using ax.scatter -> GUARANTEED 100% PERFECT CIRCLES!
    node_xs = [coords[n['id']][0] for n in nodes]
    node_ys = [coords[n['id']][1] for n in nodes]
    node_colors = [DOMAIN_COLORS[n['domain']] for n in nodes]

    # Outer white halo
    ax.scatter(node_xs, node_ys, s=850, c='#ffffff', edgecolors=[DOMAIN_COLORS[n['domain']] for n in nodes],
               linewidths=2.8, zorder=5)
    # Inner colored circle
    ax.scatter(node_xs, node_ys, s=550, c=node_colors, zorder=6)

    # Node labels inside circles
    for n in nodes:
        nid = n['id']
        px, py = coords[nid]
        lbl = n['label']
        # Node ID
        ax.text(px, py, f"S{nid}", fontsize=8.5, fontweight='bold', ha='center', va='center', color='#ffffff', zorder=7)
        # Node city label directly beneath
        ax.text(px, py - 3.2, lbl, fontsize=8, fontweight='bold', ha='center', va='top', color='#0f172a', zorder=7)

    # =========================================================================
    # BOTTOM FOOTER BANNER (IRL vs Demo Clear Distinction)
    # =========================================================================
    footer_box = FancyBboxPatch((5, 1.2), 90, 3.2, boxstyle="round,pad=0.2,rounding_size=0.8",
                                ec="#cbd5e1", fc="#f8fafc", lw=1.2, zorder=2)
    ax.add_patch(footer_box)
    
    info_text = (
        r"$\bf{REAL\ WORLD\ (Production):}$ Edge SDN Controllers run in each region; FL Server & RecurrentPPO run in central Cloud/NOC."
        "\n"
        r"$\bf{DEMO\ (Prototype):}$ High-speed analytical simulation (Gymnasium + NumPy + Yen's KSP) executed in-memory on 1 PC."
    )
    ax.text(50, 2.8, info_text, fontsize=8.5, ha='center', va='center', color='#1e293b', linespacing=1.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated clean v2 diagram: {output_path}")

if __name__ == '__main__':
    for topo in ['geant', 'nsfnet', 'abilene']:
        draw_topology_diagram(topo, f"report/figures/topology_deployment_{topo}.png")
