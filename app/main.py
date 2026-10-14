"""
Streamlit Web Demonstration & Management Console for Multi-Topology FL+RL Traffic Controller.
Provides:
1. Multi-Topology Overview (NSFNET, GEANT, Abilene) and System Diagnostics.
2. Training Console for FL (FedAdam/FedAvg) and RL (PPO) with configurable hyperparameters.
3. Live Benchmark & Evaluation Runner across Scenarios T1-T6 and Cross-Topology Transfer.
4. Visual Results Gallery and One-Click Download of Benchmark Plots, CSVs, and Trained Models.
"""

import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="FL+RL Proactive Traffic Controller",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Topologies metadata
TOPOLOGIES = {
    "nsfnet": {
        "name": "NSFNET (US Metropolitan Backbone)",
        "nodes": 14,
        "links": 44,
        "domains": 3,
        "managed_od": 20,
        "json": "data/nsfnet_topology.json",
        "tm": "data/generated/traffic_matrices.npy",
        "fl_model": "models/fl_global_model.pt",
        "rl_model": "models/ppo_agent.zip",
        "results_dir": "results/demo/nsfnet",
    },
    "geant": {
        "name": "GÉANT (Pan-European Research Mesh)",
        "nodes": 22,
        "links": 72,
        "domains": 3,
        "managed_od": 30,
        "json": "data/geant_topology.json",
        "tm": "data/generated/geant_traffic_matrices.npy",
        "fl_model": "models/fl_geant_fedadam.pt",
        "rl_model": "models/ppo_agent_geant.zip",
        "results_dir": "results/demo/geant",
    },
    "abilene": {
        "name": "Abilene (US Continental Research Trace Network)",
        "nodes": 12,
        "links": 30,
        "domains": 3,
        "managed_od": 15,
        "json": "data/abilene_topology.json",
        "tm": "data/generated/abilene_traffic_matrices.npy",
        "fl_model": "models/fl_abilene_fedadam.pt",
        "rl_model": "models/ppo_agent_abilene.zip",
        "results_dir": "results/demo/abilene",
    },
}

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-ok {
        color: #10B981;
        font-weight: 600;
    }
    .status-missing {
        color: #EF4444;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
nav_choice = st.sidebar.radio(
    "Go to",
    ["System Overview", "Interactive Training", "Benchmark & Scenarios", "Results & Downloads"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**FL+RL SDN Traffic Controller**\n\n"
    "• Proactive Congestion Avoidance\n"
    "• Multi-Domain Federated Learning (FedAdam)\n"
    "• Deep RL Multi-Path Routing (PPO)\n"
    "• Evaluated on NSFNET, GÉANT, and Abilene"
)


# ==============================================================================
# TAB 1: SYSTEM OVERVIEW
# ==============================================================================
if nav_choice == "System Overview":
    st.markdown('<div class="main-header">Proactive Traffic Engineering Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-Topology Federated Learning & Deep Reinforcement Learning for SDN Backbones</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    for i, (key, meta) in enumerate(TOPOLOGIES.items()):
        col = [col1, col2, col3][i]
        with col:
            st.markdown(f"### {meta['name']}")
            st.write(f"• **Nodes**: {meta['nodes']} PoPs")
            st.write(f"• **Directed Links**: {meta['links']} cables")
            st.write(f"• **Domains**: {meta['domains']} Administrative Partitions")
            st.write(f"• **Managed Flows**: Top {meta['managed_od']} OD Pairs")

            # Check files status
            topo_ok = (ROOT_DIR / meta["json"]).exists()
            tm_ok = (ROOT_DIR / meta["tm"]).exists()
            fl_ok = (ROOT_DIR / meta["fl_model"]).exists() or (ROOT_DIR / "models/fl_global_model.pt").exists()
            rl_ok = (ROOT_DIR / meta["rl_model"]).exists() or (ROOT_DIR / "models/ppo_agent.zip").exists()

            st.markdown(
                f"""
                <div class="metric-card">
                    <b>Artifact Status:</b><br>
                    Topology JSON: <span class="{"status-ok" if topo_ok else "status-missing"}">{"Ready" if topo_ok else "Missing"}</span><br>
                    Traffic Matrices: <span class="{"status-ok" if tm_ok else "status-missing"}">{"Ready" if tm_ok else "Not Generated"}</span><br>
                    FL Weights: <span class="{"status-ok" if fl_ok else "status-missing"}">{"Trained" if fl_ok else "Untrained"}</span><br>
                    RL Agent: <span class="{"status-ok" if rl_ok else "status-missing"}">{"Trained" if rl_ok else "Untrained"}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### 🏛 Architectural Separation: Decentralized FL vs. Centralized PPO")
    st.markdown(
        r"""
        - **Decentralized Domain FL Predictors**: Each domain edge node monitors its internal and boundary incident link byte counters. Raw NetFlow telemetry never leaves the domain perimeter. Clients train local LSTMs ($W_{in}=12, H=3$) and submit anonymous parameter updates ($\Delta w$).
        - **Adaptive FedAdam Aggregation**: The server aggregates client updates using adaptive momentum ($m_t$) and velocity ($v_t$), mitigating client drift caused by non-IID diurnal regional cycles.
        - **Central Proactive PPO Controller**: Infers over predicted near-future link loads ($t+1$ to $t+3$) and shifts heavy elephant flows across $K=3$ loop-free candidate paths **before** buffer queues saturate.
        """
    )


# ==============================================================================
# TAB 2: INTERACTIVE TRAINING
# ==============================================================================
elif nav_choice == "Interactive Training":
    st.markdown('<div class="main-header">Interactive Model Training Console</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Configure, train, and fine-tune models directly on this server</div>',
        unsafe_allow_html=True,
    )

    selected_topo = st.selectbox(
        "Select Target Topology:",
        list(TOPOLOGIES.keys()),
        format_func=lambda k: TOPOLOGIES[k]["name"],
    )
    meta = TOPOLOGIES[selected_topo]

    train_type = st.radio(
        "Select Training Subsystem:",
        ["1. Data Generation & Preprocessing", "2. Federated Learning (FL)", "3. Reinforcement Learning (PPO Agent)"],
    )

    if train_type.startswith("1."):
        st.subheader(f"Generate & Preprocess Data for {meta['name']}")
        timesteps = st.slider(
            "Timesteps to generate (5-min intervals):", min_value=200, max_value=8064, value=8064, step=288
        )
        target_mlu = st.slider("Target OSPF Peak MLU:", min_value=0.5, max_value=1.0, value=0.85, step=0.05)

        if st.button("🚀 Start Data Generation & Preprocessing", type="primary"):
            with st.spinner("Generating traffic matrices and preprocessing domain splits..."):
                cmd1 = [
                    sys.executable,
                    "-m",
                    "data.generate_tm",
                    "--topo",
                    meta["json"],
                    "--timesteps",
                    str(timesteps),
                    "--target_utilization",
                    str(target_mlu),
                ]
                cmd2 = [sys.executable, "-m", "data.preprocess", "--topology", selected_topo]

                log_area = st.empty()
                res1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=ROOT_DIR)
                res2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=ROOT_DIR)

                log_area.code(
                    f"--- STEP 1: TM GENERATION ---\n{res1.stdout}\n{res1.stderr}\n\n--- STEP 2: PREPROCESSING ---\n{res2.stdout}\n{res2.stderr}"
                )
                st.success("Data pipeline executed successfully!")

    elif train_type.startswith("2."):
        st.subheader(f"Train Federated Learning Model for {meta['name']}")
        col_fl1, col_fl2 = st.columns(2)
        with col_fl1:
            fl_strategy = st.selectbox("Aggregation Strategy:", ["fedadam", "fedavg"], index=0)
            fl_rounds = st.slider("Number of FL Rounds:", min_value=2, max_value=50, value=10, step=2)
        with col_fl2:
            model_out_path = st.text_input("Model Save Path:", meta["fl_model"])

        if st.button("🧠 Launch Federated Training", type="primary"):
            cmd = [
                sys.executable,
                "-m",
                "fl.train_fl",
                "--topology",
                selected_topo,
                "--strategy",
                fl_strategy,
                "--rounds",
                str(fl_rounds),
                "--model_save_path",
                model_out_path,
            ]
            st.info(f"Executing: `{' '.join(cmd)}`")
            with st.spinner("Training FL Model..."):
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
                st.code(proc.stdout if proc.stdout else proc.stderr)
                if proc.returncode == 0:
                    st.success("FL Training completed and model saved!")
                else:
                    st.error(f"FL Training failed (Code {proc.returncode}).")

    else:
        st.subheader(f"Train PPO Routing Agent for {meta['name']}")
        col_rl1, col_rl2 = st.columns(2)
        with col_rl1:
            rl_timesteps = st.select_slider(
                "Training Timesteps:",
                options=[1000, 5000, 20000, 50000, 100000, 200000],
                value=20000,
            )
            rl_envs = st.slider("Parallel Gym Environments (n_envs):", min_value=1, max_value=16, value=4, step=1)
        with col_rl2:
            surge_p = st.slider("Surge Anomaly Probability:", min_value=0.0, max_value=0.3, value=0.15, step=0.05)
            custom_od = st.number_input("Managed OD Pairs (0 = default):", min_value=0, max_value=50, value=0)

        if st.button("🎯 Launch PPO Training", type="primary"):
            cmd = [
                sys.executable,
                "-m",
                "rl.train_rl",
                "--topology",
                selected_topo,
                "--timesteps",
                str(rl_timesteps),
                "--n_envs",
                str(rl_envs),
            ]
            if custom_od > 0:
                cmd.extend(["--num_managed_od", str(custom_od)])

            st.info(f"Executing: `{' '.join(cmd)}`")
            with st.spinner(f"Training PPO Agent for {rl_timesteps} steps on {selected_topo}..."):
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
                st.code(proc.stdout if proc.stdout else proc.stderr)
                if proc.returncode == 0:
                    st.success("PPO Training completed and agent saved!")
                else:
                    st.error(f"PPO Training failed (Code {proc.returncode}).")


# ==============================================================================
# TAB 3: BENCHMARK & SCENARIOS
# ==============================================================================
elif nav_choice == "Benchmark & Scenarios":
    st.markdown('<div class="main-header">Evaluation Demo & Benchmark Runner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Run multi-scenario simulations comparing OSPF, Random, and Trained Policies</div>',
        unsafe_allow_html=True,
    )

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        eval_topo = st.selectbox(
            "Topology to Benchmark:",
            ["all", "nsfnet", "geant", "abilene"],
            index=0,
            format_func=lambda x: "All 3 Topologies (Sequential)" if x == "all" else TOPOLOGIES[x]["name"],
        )
        cross_eval_opt = st.checkbox("Cross-Topology Transfer (Use GÉANT model on Abilene)", value=False)
    with col_b2:
        eval_steps = st.slider("Evaluation Steps per Episode:", min_value=20, max_value=200, value=100, step=20)
        skip_random = st.checkbox("Skip Untrained (Random) Policy", value=True)

    if st.button("📊 Run Multi-Scenario Benchmark", type="primary"):
        cmd = [
            sys.executable,
            "-m",
            "scripts.demo",
            "--topology",
            eval_topo,
            "--steps",
            str(eval_steps),
        ]
        if cross_eval_opt:
            cmd.append("--cross_eval")
        if skip_random:
            cmd.append("--no_random")

        st.info(f"Running benchmark: `{' '.join(cmd)}`")
        with st.spinner("Running simulation across seeds and scenarios (T1-T6)..."):
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
            st.code(proc.stdout if proc.stdout else proc.stderr)
            if proc.returncode == 0:
                st.success("Benchmark completed! Check results in the 'Results & Downloads' tab.")
            else:
                st.error("Benchmark encountered an issue.")


# ==============================================================================
# TAB 4: RESULTS & DOWNLOADS
# ==============================================================================
else:
    st.markdown('<div class="main-header">Results Gallery & Downloads</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inspect generated visualization timelines, summary metrics, and export artifacts</div>',
        unsafe_allow_html=True,
    )

    view_topo = st.selectbox("Select Topology Results:", ["nsfnet", "geant", "abilene"])
    target_dir = ROOT_DIR / "results" / "demo" / view_topo
    if not target_dir.exists() and view_topo == "nsfnet":
        target_dir = ROOT_DIR / "results" / "demo"

    # 1. Image Gallery
    st.subheader("📈 Visualization Charts")
    png_files = sorted(list(target_dir.glob("*.png"))) if target_dir.exists() else []

    if png_files:
        selected_plot = st.selectbox("Select Chart to View:", [p.name for p in png_files])
        img_path = target_dir / selected_plot
        st.image(str(img_path), caption=selected_plot, use_column_width=True)

        with st.expander("Show All Generated Plots in Carousel"):
            for p in png_files:
                st.markdown(f"#### `{p.name}`")
                st.image(str(p), use_column_width=True)
    else:
        st.warning(f"No generated PNG plots found in `{target_dir}`. Please run the benchmark demo first.")

    # 2. CSV Benchmark Tables
    st.markdown("---")
    st.subheader("📋 Benchmark Summary Tables")
    csv_candidates = [
        ROOT_DIR / "results" / "summary_table.csv",
        ROOT_DIR / "results" / "benchmark_results.csv",
        ROOT_DIR / "results" / f"fl_{view_topo}_prediction_metrics.csv",
        ROOT_DIR / "results" / "fl_prediction_metrics.csv",
    ]

    for csv_file in csv_candidates:
        if csv_file.exists():
            st.markdown(f"**`{csv_file.name}`**")
            try:
                df = pd.read_csv(csv_file)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.write(f"Could not render CSV: {e}")

    # 3. Download Artifacts
    st.markdown("---")
    st.subheader("📦 Download Reports & Model Weights")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.markdown("#### 1. Download All Results (ZIP)")
        # Create in-memory zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            results_path = ROOT_DIR / "results"
            if results_path.exists():
                for root, _, files in os.walk(results_path):
                    for file in files:
                        fp = Path(root) / file
                        zf.write(fp, arcname=fp.relative_to(ROOT_DIR))

        st.download_button(
            label="📥 Download Results ZIP (Plots & CSVs)",
            data=zip_buffer.getvalue(),
            file_name="traffic_controller_results.zip",
            mime="application/zip",
        )

    with col_dl2:
        st.markdown("#### 2. Download Model Artifacts")
        model_files = list((ROOT_DIR / "models").glob("*.*"))
        for mf in model_files:
            if mf.is_file():
                with open(mf, "rb") as f:
                    st.download_button(
                        label=f"💾 Download {mf.name} ({mf.stat().st_size / 1024:.1f} KB)",
                        data=f.read(),
                        file_name=mf.name,
                        key=f"dl_{mf.name}",
                    )
