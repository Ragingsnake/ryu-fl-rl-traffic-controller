import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set high-DPI and modern styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(15, 9.5), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 9.5)
ax.axis('off')

# Color palette
c_telemetry = "#F1F5F9"      # Slate light
c_telemetry_b = "#64748B"    # Slate border
c_fl = "#EFF6FF"             # Blue light
c_fl_b = "#2563EB"           # Blue border
c_fusion = "#FEF3C7"         # Amber light
c_fusion_b = "#D97706"       # Amber border
c_ppo = "#F0FDF4"            # Green light
c_ppo_b = "#16A34A"          # Green border
c_sdn = "#FAF5FF"            # Purple light
c_sdn_b = "#9333EA"          # Purple border
c_text_dark = "#0F172A"

def draw_box(ax, x, y, w, h, bg_color, border_color, title, subtitle="", radius=0.2, lw=2.0):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius},rounding_size=0.2",
        linewidth=lw, edgecolor=border_color, facecolor=bg_color, zorder=2
    )
    ax.add_patch(rect)
    if title:
        ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center',
                fontsize=11.5, fontweight='bold', color=c_text_dark, zorder=3)
    if subtitle:
        ax.text(x + w/2, y + h - 0.75, subtitle, ha='center', va='center',
                fontsize=9.5, fontstyle='italic', color="#475569", zorder=3)
    return rect

def draw_sub_box(ax, x, y, w, h, bg_color, border_color, text, fontsize=9, bold=False):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.1",
        linewidth=1.2, edgecolor=border_color, facecolor=bg_color, zorder=4
    )
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=c_text_dark, zorder=5)
    return rect

# Title Banner
ax.text(7.5, 9.15, "KIẾN TRÚC TÍCH HỢP ĐIỀU KHIỂN LƯU LƯỢNG CHỦ ĐỘNG: FL-LSTM VÀ RECURRENT-PPO",
        ha='center', va='center', fontsize=14, fontweight='bold', color="#1E293B")
ax.text(7.5, 8.8, "Cơ chế nạp dự báo lưu lượng (State Injection) & Vòng lặp tối ưu hóa môi trường Gymnasium",
        ha='center', va='center', fontsize=10.5, fontstyle='italic', color="#64748B")

# ==========================================
# 1. PHYSICAL & TELEMETRY PLANE (Bottom-Left)
# ==========================================
draw_box(ax, 0.6, 0.6, 4.4, 2.5, c_telemetry, c_telemetry_b,
         "1. TẦNG VẬT LÝ & THU THẬP DỮ LIỆU", "Physical Telemetry & Ingestion")

draw_sub_box(ax, 0.9, 1.7, 3.8, 0.7, "#FFFFFF", c_telemetry_b,
             "Mạng WAN Thực Tế (GÉANT / Abilene / NSFNET)\nGiám sát sFlow / In-band Network Telemetry", fontsize=8.5)
draw_sub_box(ax, 0.9, 0.8, 3.8, 0.65, "#FFFFFF", c_telemetry_b,
             "Trích xuất đặc trưng mạng:\n• Tải liên kết hiện tại: u_t ∈ ℝ^|E|\n• Nhu cầu luồng lớn (Heavy-Hitters): d_t ∈ ℝ^|M|", fontsize=8)

# ==========================================
# 2. FL-LSTM PREDICTION PLANE (Top-Left)
# ==========================================
draw_box(ax, 0.6, 4.0, 4.4, 4.4, c_fl, c_fl_b,
         "2. DỰ BÁO LƯU LƯỢNG: FL-LSTM", "TrafficLSTM Predictor (Lookahead = 45 phút)")

draw_sub_box(ax, 0.9, 6.9, 3.8, 0.85, "#FFFFFF", c_fl_b,
             "Bộ đệm lịch sử tải (Sliding Window):\nX_t = [u_{t-11}, ..., u_t] ∈ ℝ^{B × 12 × F_in}\n(Lookback T=12 bước = 3 giờ)", fontsize=8.5)

draw_sub_box(ax, 0.9, 5.4, 3.8, 1.25, "#FFFFFF", c_fl_b,
             "Kiến trúc Mạng TrafficLSTM:\n• Layer 1: LSTM (hidden=64, batch_first=True)\n• Dropout: p = 0.2 (chống quá khớp)\n• Layer 2: LSTM (hidden=64)\n• Linear Head: 64 → (H × F_in), H=3", fontsize=8.5)

draw_sub_box(ax, 0.9, 4.25, 3.8, 0.85, "#DBEAFE", c_fl_b,
             "Máy chủ FL Server (Tổng hợp FedAdam):\n• Học phân tán bảo mật quyền riêng tư 3 miền\n• Sai số MAE < 0.05, không rò rỉ dữ liệu gốc", fontsize=8)

# ==========================================
# 3. STATE INJECTION FUSION (Center)
# ==========================================
draw_box(ax, 5.6, 3.5, 3.8, 4.9, c_fusion, c_fusion_b,
         "3. THIẾT LẬP MÔI TRƯỜNG TRẠNG THÁI", "State Space Injection (Concatenation)")

ax.text(7.5, 7.3, "CƠ CHẾ 'GÍ' DỰ BÁO VÀO PPO", ha='center', va='center',
        fontsize=10, fontweight='bold', color="#B45309", zorder=3)
ax.text(7.5, 6.95, "Đóng gói vector quan sát Gymnasium:", ha='center', va='center',
        fontsize=8.5, fontstyle='italic', color="#78350F", zorder=3)

draw_sub_box(ax, 5.85, 5.7, 3.3, 0.95, "#FEF9C3", c_fusion_b,
             "Vector Dự Báo Tương Lai (t+1)\nû_{t+1} ∈ ℝ^|E|\n(Tải liên kết dự phóng từ FL-LSTM)", fontsize=8.5, bold=True)

draw_sub_box(ax, 5.85, 4.6, 3.3, 0.85, "#FEF9C3", c_fusion_b,
             "Tải Thực Tế Hiện Tại: u_t ∈ ℝ^|E|\nNhu Cầu Luồng Lớn: d_t ∈ ℝ^|M|", fontsize=8.5)

draw_sub_box(ax, 5.85, 3.7, 3.3, 0.7, "#FEF3C7", c_fusion_b,
             "Observation Vector Hoàn Chỉnh:\ns_t = [ u_t  ||  û_{t+1}  ||  d_t ]\ndim(s_t) = 2|E| + |M|", fontsize=9, bold=True)

# ==========================================
# 4. RECURRENTPPO AGENT (Top-Right)
# ==========================================
draw_box(ax, 10.0, 4.5, 4.4, 3.9, c_ppo, c_ppo_b,
         "4. TÁC TỬ ĐIỀU KHIỂN: RecurrentPPO", "MlpLstmPolicy với Quán Tính Hành Động")

draw_sub_box(ax, 10.3, 6.8, 3.8, 0.9, "#FFFFFF", c_ppo_b,
             "Tầng Nhớ Trạng Thái (Action Inertia):\n• LSTM Cell (hidden_size = 128)\n• Duy trì h_{t-1}, c_{t-1} triệt tiêu rung giật (flapping)", fontsize=8.5)

draw_sub_box(ax, 10.3, 5.65, 1.8, 0.9, "#DCFCE7", c_ppo_b,
             "Actor Head π_θ\nMLP (256→128→K)\nPhân phối xác suất\nchọn tuyến", fontsize=8)

draw_sub_box(ax, 12.3, 5.65, 1.8, 0.9, "#DCFCE7", c_ppo_b,
             "Critic Head V_ϕ\nMLP (256→128→1)\nƯớc lượng giá trị\nphần thưởng V(s_t)", fontsize=8)

draw_sub_box(ax, 10.3, 4.75, 3.8, 0.7, "#DCFCE7", c_ppo_b,
             "Hành Động Đa Rời Rạc (MultiDiscrete):\na_t = [a_t^(1), ..., a_t^(|M|)], a_t^(m) ∈ {0, 1, 2}\n(K=3 đường ngắn nhất Yen cho mỗi luồng)", fontsize=8, bold=True)

# ==========================================
# 5. SDN & CLOSED LOOP (Bottom-Right)
# ==========================================
draw_box(ax, 10.0, 0.6, 4.4, 3.2, c_sdn, c_sdn_b,
         "5. BỘ ĐIỀU KHIỂN SDN & PHẢN HỒI", "Ryu SDN Controller & Reward Evaluation")

draw_sub_box(ax, 10.3, 2.2, 3.8, 0.8, "#FFFFFF", c_sdn_b,
             "SDN Controller & Bộ Lọc An Toàn:\n• Projected Load Verification (Kiểm tra tải an toàn)\n• Hold-Down Damping (Giữ tuyến tối thiểu 15 bước)\n• Cài đặt OpenFlow FlowMod Rules", fontsize=8)

draw_sub_box(ax, 10.3, 0.8, 3.8, 1.15, "#F3E8FF", c_sdn_b,
             "Đánh Giá Môi Trường & Hàm Phần Thưởng:\nR_t = - [ 1.0·MLU + 0.5·Delay_norm\n         + w_excess·ExcessDelay + 40·Loss + 0.08·Churn ]\n(Mô hình hàng đợi viễn thông M/M/1)", fontsize=8, bold=True)

# ==========================================
# ARROWS & DATA CONNECTIONS
# ==========================================
arrow_kw = dict(arrowstyle="-|>", color="#1E293B", lw=2, mutation_scale=16, zorder=6)
arrow_fl = dict(arrowstyle="-|>", color="#2563EB", lw=2.5, mutation_scale=18, zorder=6)
arrow_action = dict(arrowstyle="-|>", color="#16A34A", lw=2.2, mutation_scale=16, zorder=6)
arrow_fb = dict(arrowstyle="-|>", color="#DC2626", lw=2, ls="--", mutation_scale=16, zorder=6)

# Telemetry -> FL
ax.annotate("", xy=(2.8, 4.0), xytext=(2.8, 3.1), arrowprops=arrow_kw)
ax.text(2.9, 3.5, "Chuỗi lịch sử\nu_t nạp vào đệm", fontsize=8, color="#475569")

# Telemetry -> Fusion
ax.annotate("", xy=(5.6, 4.8), xytext=(4.4, 2.8),
            arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.8, mutation_scale=14,
                            connectionstyle="arc3,rad=-0.15"))
ax.text(4.4, 3.8, "Tải hiện tại u_t\n& Nhu cầu d_t", fontsize=8, color="#475569")

# FL Predictor -> Fusion (THE KEY INJECTION ARROW!)
ax.annotate("", xy=(5.6, 6.2), xytext=(5.0, 6.2), arrowprops=arrow_fl)
ax.text(4.8, 6.45, "û_{t+1}", fontsize=11, fontweight='bold', color="#2563EB")

# Fusion -> PPO Agent
ax.annotate("", xy=(10.0, 6.5), xytext=(9.4, 6.5), arrowprops=arrow_kw)
ax.text(9.45, 6.75, "s_t", fontsize=11, fontweight='bold', color="#D97706")

# PPO Agent -> SDN Controller
ax.annotate("", xy=(12.2, 3.8), xytext=(12.2, 4.5), arrowprops=arrow_action)
ax.text(12.35, 4.15, "Hành động\na_t (K=3)", fontsize=8.5, fontweight='bold', color="#16A34A")

# SDN Controller -> Telemetry (Execution in network)
ax.annotate("", xy=(5.0, 1.3), xytext=(10.0, 1.3),
            arrowprops=dict(arrowstyle="-|>", color="#9333EA", lw=2, mutation_scale=16))
ax.text(7.5, 1.45, "Thực thi cấu hình OpenFlow trên Switch", ha='center', fontsize=8.5, fontweight='bold', color="#9333EA")

# Closed Loop Reward Feedback
ax.annotate("", xy=(10.0, 7.6), xytext=(10.0, 2.0),
            arrowprops=dict(arrowstyle="-|>", color="#DC2626", lw=2, ls="--", mutation_scale=16,
                            connectionstyle="arc3,rad=-0.35"))
ax.text(8.8, 2.5, "Phản hồi Reward R_t\n& Trạng thái kế s_{t+1}", ha='center', fontsize=8.5, fontweight='bold', color="#DC2626")

plt.tight_layout()
output_path = "report/figures/fig_lstm_ppo_env.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated {output_path}")
