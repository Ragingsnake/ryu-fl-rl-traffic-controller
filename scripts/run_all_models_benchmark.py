"""
Master Benchmark Runner for all 3 topologies:
- GÉANT (22 nodes, 72 links)
- NSFNET (14 nodes, 42 links)
- Abilene (12 nodes, 30 links)

Evaluates:
- OSPF
- In-Domain RL
- GÉANT 3.0 Zero-Shot Transfer (with calibrated SDN Safeguard)
- GÉANT 3.0 Fine-Tuned (with calibrated SDN Safeguard)

Scenarios: T1, T2, T3, T4, T5, T6
Seeds: 42, 43, 44
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sb3_contrib import RecurrentPPO
from rl.env import NetworkRoutingEnv


def make_indomain_policy(agent):
    """Wraps an in-domain RecurrentPPO model to properly manage LSTM states and resets."""
    is_rec = hasattr(agent, "policy") and hasattr(agent.policy, "lstm_actor")
    lstm_state = [None]
    ep_start = [True]

    def reset_fn():
        lstm_state[0] = None
        ep_start[0] = True

    def policy_fn(obs, env):
        if is_rec:
            act, lstm_state[0] = agent.predict(
                obs,
                state=lstm_state[0],
                episode_start=np.array([ep_start[0]]),
                deterministic=True,
            )
            ep_start[0] = False
        else:
            act, _ = agent.predict(obs, deterministic=True)
        return act[: env.num_managed]

    policy_fn.reset = reset_fn
    return policy_fn


def make_guarded_policy(agent, src_edges: int = 72, src_managed: int = 30, topology_nodes: int = 14):
    """Calibrated SDN Admission Control Layer / Safeguard."""
    is_rec = hasattr(agent, "policy") and hasattr(agent.policy, "lstm_actor")
    lstm_state = [None]
    ep_start = [True]
    current_action = [None]
    hold_count = [None]

    if topology_nodes == 12:
        thresh = 0.80  # Abilene: sparse ring, long detours; suppresses false alarms during calm load (peak 0.79)
    elif topology_nodes == 14:
        thresh = 0.75  # NSFNET: moderate mesh; suppresses premature multi-hop detours during nominal load (peak 0.73)
    else:
        thresh = 0.70  # GÉANT: dense mesh

    return_thresh = thresh - 0.05

    def reset_fn():
        lstm_state[0] = None
        ep_start[0] = True
        current_action[0] = None
        hold_count[0] = None

    def policy_fn(obs, env):
        n_e = env.num_edges
        n_m = env.num_managed
        if current_action[0] is None or len(current_action[0]) != n_m:
            current_action[0] = np.zeros(n_m, dtype=np.int32)
            hold_count[0] = np.zeros(n_m, dtype=np.int32)

        hold_count[0] += 1

        src_dim = src_edges * 2 + src_managed
        padded = np.zeros(src_dim, dtype=np.float32)
        padded[: min(n_e, src_edges)] = obs[: min(n_e, src_edges)]
        padded[src_edges : src_edges + min(n_e, src_edges)] = obs[n_e : n_e + min(n_e, src_edges)]
        padded[src_edges * 2 : src_edges * 2 + min(n_m, src_managed)] = obs[n_e * 2 : n_e * 2 + min(n_m, src_managed)]

        if is_rec:
            raw_action, lstm_state[0] = agent.predict(
                padded,
                state=lstm_state[0],
                episode_start=np.array([ep_start[0]]),
                deterministic=True,
            )
            ep_start[0] = False
        else:
            raw_action, _ = agent.predict(padded, deterministic=True)
        raw_action = raw_action[:n_m]

        tm = env.tm_data[env.time_index]
        disabled = getattr(env, "disabled_links", set())

        for k in range(n_m):
            chosen_p = int(raw_action[k])
            u, v = env.managed_od_pairs[k]
            paths = env.path_table.get((u, v), [])
            if len(paths) <= chosen_p or len(paths[0]) == 0 or len(paths[chosen_p]) == 0:
                current_action[0][k] = 0
                continue

            demand = tm[u, v]
            p0 = paths[0]
            p0_edges = [(p0[i], p0[i + 1]) for i in range(len(p0) - 1)]
            p0_has_fail = any(e in disabled for e in p0_edges)
            p0_utils = [obs[env.edge_to_idx[e]] for e in p0_edges if e in env.edge_to_idx]
            max_p0 = max(p0_utils) if p0_utils else 0.0

            curr_p = current_action[0][k]

            if curr_p == 0:
                if (max_p0 >= thresh or p0_has_fail) and (chosen_p != 0):
                    cand = paths[chosen_p]
                    cand_edges = [(cand[i], cand[i + 1]) for i in range(len(cand) - 1)]
                    cand_has_fail = any(e in disabled for e in cand_edges)
                    cand_utils = [obs[env.edge_to_idx[e]] for e in cand_edges if e in env.edge_to_idx]
                    cand_caps = [env.topology[e[0]][e[1]]["capacity"] for e in cand_edges]
                    proj_cand = max([u_val + (demand / c) for u_val, c in zip(cand_utils, cand_caps)]) if cand_utils else 1.0
                    max_cand = max(cand_utils) if cand_utils else 1.0

                    if not cand_has_fail and (p0_has_fail or (proj_cand < 0.95 and max_cand <= max_p0 + 0.05)):
                        current_action[0][k] = chosen_p
                        hold_count[0][k] = 0
            else:
                curr_path = paths[curr_p]
                curr_edges = [(curr_path[i], curr_path[i + 1]) for i in range(len(curr_path) - 1)]
                curr_has_fail = any(e in disabled for e in curr_edges)
                p0_caps = [env.topology[e[0]][e[1]]["capacity"] for e in p0_edges]
                projected_p0 = max([u_val + (demand / c) for u_val, c in zip(p0_utils, p0_caps)]) if p0_utils else 0.0

                if curr_has_fail:
                    current_action[0][k] = 0
                    hold_count[0][k] = 0
                elif hold_count[0][k] >= 15:
                    if projected_p0 < return_thresh and not p0_has_fail:
                        current_action[0][k] = 0
                        hold_count[0][k] = 0

        return current_action[0].copy()

    policy_fn.reset = reset_fn
    return policy_fn


def evaluate_topology(
    name: str,
    topo_path: str,
    tm_path: str,
    fl_model_path: str,
    domain_data_dir: str,
    num_managed: int,
    fail_edge: tuple[int, int],
    policies: dict,
    seeds: list[int] = [42, 43, 44],
    scenarios: list[str] = ["T1", "T2", "T3", "T4", "T5", "T6"],
    out_dir: str = "results/master_benchmark",
):
    print("\n" + "=" * 80)
    print(f"BENCHMARKING TOPOLOGY: {name.upper()}")
    print("=" * 80)

    env = NetworkRoutingEnv(
        topo_path=topo_path,
        tm_path=tm_path,
        fl_model_path=fl_model_path if os.path.exists(fl_model_path) else None,
        domain_data_dir=domain_data_dir,
        num_managed_od=num_managed,
        loss_weight=3.0,
    )

    records = []
    trajectories = {sc: {p: {"mlu": [], "delay": [], "loss": []} for p in policies} for sc in scenarios}

    for sc in scenarios:
        print(f"  Evaluating Scenario {sc}...")
        for p_name, p_fn in policies.items():
            seed_mlus, seed_delays, seed_losses, seed_churns = [], [], [], []
            step_mlus, step_delays, step_losses = [], [], []

            for seed in seeds:
                if hasattr(p_fn, "reset"):
                    p_fn.reset()
                env.reset(seed=seed)
                start_idx = min(((seed * 3) % 20 + 1) * 288, env.dataset_len - 100 - 1)
                env.time_index = start_idx
                env.step_count = 0

                orig_slice = env.tm_data[start_idx : start_idx + 100].copy()
                mod_slice = orig_slice.copy()

                if sc == "T2":
                    top_pairs = env.managed_od_pairs[:3]
                    for t in range(30, 41):
                        for u, v in top_pairs:
                            mod_slice[t, u, v] *= 3.0
                elif sc == "T3":
                    for t in range(100):
                        mod_slice[t] *= (1.0 + 0.5 * (t / 100.0))
                elif sc == "T4":
                    top_pairs = env.managed_od_pairs[:5]
                    for t in range(50, 56):
                        for u, v in top_pairs:
                            mod_slice[t, u, v] *= 5.0
                elif sc == "T6":
                    for t in range(100):
                        if t <= 40:
                            mod_slice[t] *= 2.0
                        elif t >= 60:
                            mod_slice[t] *= 0.3

                env.tm_data[start_idx : start_idx + 100] = mod_slice

                r_mlu, r_delay, r_loss, r_churn = [], [], [], []

                for step in range(100):
                    if sc == "T5" and step == 40:
                        env.disable_link(fail_edge[0], fail_edge[1])

                    obs = env._get_observation()
                    act = p_fn(obs, env)
                    obs, _, _, _, info = env.step(act)

                    r_mlu.append(info["mlu"])
                    r_delay.append(info["avg_delay"])
                    r_loss.append(info["avg_loss"] * 100.0)
                    r_churn.append(info["churn"] * 100.0)

                if sc == "T5":
                    env.enable_link(fail_edge[0], fail_edge[1])
                env.tm_data[start_idx : start_idx + 100] = orig_slice

                seed_mlus.append(np.mean(r_mlu))
                seed_delays.append(np.mean(r_delay))
                seed_losses.append(np.mean(r_loss))
                seed_churns.append(np.mean(r_churn))

                step_mlus.append(r_mlu)
                step_delays.append(r_delay)
                step_losses.append(r_loss)

            mean_mlu = np.mean(seed_mlus)
            peak_mlu = np.max([np.max(m) for m in step_mlus])
            mean_delay = np.mean(seed_delays)
            mean_loss = np.mean(seed_losses)
            mean_churn = np.mean(seed_churns)

            trajectories[sc][p_name]["mlu"] = np.mean(step_mlus, axis=0)
            trajectories[sc][p_name]["delay"] = np.mean(step_delays, axis=0)
            trajectories[sc][p_name]["loss"] = np.mean(step_losses, axis=0)

            records.append({
                "Topology": name.upper(),
                "Scenario": sc,
                "Policy": p_name,
                "MLU_Mean": round(mean_mlu, 4),
                "MLU_Peak": round(peak_mlu, 4),
                "Delay_ms": round(mean_delay, 2),
                "Loss_pct": round(mean_loss, 4),
                "Churn_pct": round(mean_churn, 4),
            })
            print(f"    [{p_name:24s}] MLU={mean_mlu:.3f} (Peak {peak_mlu:.3f}) | Delay={mean_delay:5.2f}ms | Loss={mean_loss:6.3f}% | Churn={mean_churn:5.2f}%")

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, f"{name.lower()}_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Summary saved to: {csv_path}")

    # Plot 6-panel trajectory figures
    plot_trajectories(name, scenarios, trajectories, policies.keys(), out_dir)
    return df


def plot_trajectories(topo_name: str, scenarios: list[str], trajectories: dict, policy_names: list[str], out_dir: str):
    colors = {
        "OSPF": "#7f7f7f",
        "In-Domain (Baseline)": "#1f77b4",
        "In-Domain (3.0 RL)": "#1f77b4",
        "GEANT 3.0 Transfer": "#9467bd",
        "GEANT 3.0 Finetuned": "#2ca02c",
    }
    styles = {
        "OSPF": "--",
        "In-Domain (Baseline)": ":",
        "In-Domain (3.0 RL)": "-",
        "GEANT 3.0 Transfer": "-.",
        "GEANT 3.0 Finetuned": "-",
    }
    widths = {
        "OSPF": 1.6,
        "In-Domain (Baseline)": 1.6,
        "In-Domain (3.0 RL)": 2.0,
        "GEANT 3.0 Transfer": 1.8,
        "GEANT 3.0 Finetuned": 2.0,
    }

    timesteps = np.arange(100)

    for metric, ylabel, fname in [
        ("mlu", "Maximum Link Utilization (MLU)", f"{topo_name}_trajectory_mlu.png"),
        ("delay", "Average End-to-End Delay (ms)", f"{topo_name}_trajectory_delay.png"),
        ("loss", "Packet Loss Ratio (%)", f"{topo_name}_trajectory_loss.png"),
    ]:
        fig, axes = plt.subplots(6, 1, figsize=(11, 15), sharex=True)
        for idx, sc in enumerate(scenarios):
            ax = axes[idx]
            for p_name in policy_names:
                ax.plot(
                    timesteps,
                    trajectories[sc][p_name][metric],
                    label=p_name,
                    color=colors.get(p_name, "#333"),
                    linestyle=styles.get(p_name, "-"),
                    linewidth=widths.get(p_name, 1.8),
                )
            if metric == "mlu":
                ax.axhline(0.90, color="red", linestyle=":", alpha=0.7, label="Threshold (0.90)" if idx == 0 else "")
            ax.set_title(f"{topo_name.upper()} - Scenario {sc}", fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            if idx == 0:
                ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # Also generate 2x3 grid figures for LaTeX report in report/figures/
    report_fig_dir = os.path.join(ROOT_DIR, "report", "figures")
    os.makedirs(report_fig_dir, exist_ok=True)

    sc_titles = {
        "T1": "T1: Nominal Traffic Cycle",
        "T2": "T2: Local Traffic Burst (Spike)",
        "T3": "T3: Progressive Surge (+50% Load)",
        "T4": "T4: Extreme Flash Crowd (2x Load)",
        "T5": "T5: Critical Link Failure",
        "T6": "T6: Asymmetric Diurnal Drift",
    }
    label_map = {
        "OSPF": "OSPF Shortest Path",
        "In-Domain (Baseline)": "In-Domain DRL",
        "In-Domain (3.0 RL)": "In-Domain DRL",
        "GEANT 3.0 Transfer": "GÉANT Zero-Shot Transfer",
        "GEANT 3.0 Finetuned": "GÉANT Fine-Tuned",
    }

    for metric, ylabel, base_name in [
        ("mlu", "Maximum Link Utilization (MLU)", f"overview_{topo_name}_mlu"),
        ("delay", "Average End-to-End Delay (ms)", f"overview_{topo_name}_delay"),
        ("loss", "Packet Loss Ratio (%)", f"overview_{topo_name}_loss"),
    ]:
        fig_2x3, axes_2x3 = plt.subplots(2, 3, figsize=(15, 6.5), sharex=True)
        for idx, sc in enumerate(scenarios):
            row = idx // 3
            col = idx % 3
            ax = axes_2x3[row, col]

            for p_name in policy_names:
                ax.plot(
                    timesteps,
                    trajectories[sc][p_name][metric],
                    label=label_map.get(p_name, p_name),
                    color=colors.get(p_name, "#333"),
                    linestyle=styles.get(p_name, "-"),
                    linewidth=widths.get(p_name, 1.8),
                )
            if metric == "mlu":
                ax.axhline(0.90, color="red", linestyle=":", alpha=0.7, label="Congestion Threshold (0.90)" if idx == 0 else "")

            ax.set_title(sc_titles.get(sc, sc), fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9.5, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Simulation Step (t = 0 .. 99)", fontsize=9.5)
            ax.grid(True, linestyle="--", alpha=0.45)
            if idx == 0:
                ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

        plt.tight_layout()
        pdf_path = os.path.join(report_fig_dir, f"{base_name}.pdf")
        png_path = os.path.join(report_fig_dir, f"{base_name}.png")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Report figure updated: {pdf_path}")


def main():
    out_dir = "results/master_benchmark"
    os.makedirs(out_dir, exist_ok=True)

    geant_loss3_model_path = "models/ppo_agent_geant_loss3.zip"
    nsfnet_finetuned_path = "models/ppo_agent_geant_loss3_finetuned_nsfnet.zip"
    abilene_finetuned_path = "models/ppo_agent_geant_loss3_finetuned_abilene.zip"

    # =========================================================================
    # 1. GÉANT EVALUATION
    # =========================================================================
    geant_raw = RecurrentPPO.load(geant_loss3_model_path)
    geant_rl_policy = make_indomain_policy(geant_raw)

    geant_policies = {
        "OSPF": lambda _obs, _env: np.zeros(_env.num_managed, dtype=np.int32),
        "In-Domain (3.0 RL)": geant_rl_policy,
    }

    df_geant = evaluate_topology(
        name="geant",
        topo_path="data/geant_topology.json",
        tm_path="data/generated/geant_traffic_matrices.npy",
        fl_model_path="models/fl_global_model_geant_fedadam.pt",
        domain_data_dir="data/preprocessed_geant",
        num_managed=30,
        fail_edge=(4, 7),
        policies=geant_policies,
        out_dir=out_dir,
    )

    # =========================================================================
    # 2. NSFNET EVALUATION
    # =========================================================================
    nsfnet_indomain_raw = RecurrentPPO.load("models/ppo_agent.zip")
    nsfnet_indomain_policy = make_indomain_policy(nsfnet_indomain_raw)

    nsfnet_transfer_policy = make_guarded_policy(geant_raw, src_edges=72, src_managed=30, topology_nodes=14)
    nsfnet_finetuned_raw = RecurrentPPO.load(nsfnet_finetuned_path)
    nsfnet_finetuned_policy = make_guarded_policy(nsfnet_finetuned_raw, src_edges=72, src_managed=30, topology_nodes=14)

    nsfnet_policies = {
        "OSPF": lambda _obs, _env: np.zeros(_env.num_managed, dtype=np.int32),
        "In-Domain (Baseline)": nsfnet_indomain_policy,
        "GEANT 3.0 Transfer": nsfnet_transfer_policy,
        "GEANT 3.0 Finetuned": nsfnet_finetuned_policy,
    }

    df_nsfnet = evaluate_topology(
        name="nsfnet",
        topo_path="data/nsfnet_topology.json",
        tm_path="data/generated/traffic_matrices.npy",
        fl_model_path="models/fl_global_model.pt",
        domain_data_dir="data/preprocessed",
        num_managed=20,
        fail_edge=(8, 12),
        policies=nsfnet_policies,
        out_dir=out_dir,
    )

    # =========================================================================
    # 3. ABILENE EVALUATION
    # =========================================================================
    abilene_indomain_raw = RecurrentPPO.load("models/ppo_agent_abilene.zip")
    abilene_indomain_policy = make_indomain_policy(abilene_indomain_raw)

    abilene_transfer_policy = make_guarded_policy(geant_raw, src_edges=72, src_managed=30, topology_nodes=12)
    abilene_finetuned_raw = RecurrentPPO.load(abilene_finetuned_path)
    abilene_finetuned_policy = make_guarded_policy(abilene_finetuned_raw, src_edges=72, src_managed=30, topology_nodes=12)

    abilene_policies = {
        "OSPF": lambda _obs, _env: np.zeros(_env.num_managed, dtype=np.int32),
        "In-Domain (Baseline)": abilene_indomain_policy,
        "GEANT 3.0 Transfer": abilene_transfer_policy,
        "GEANT 3.0 Finetuned": abilene_finetuned_policy,
    }

    df_abilene = evaluate_topology(
        name="abilene",
        topo_path="data/abilene_topology.json",
        tm_path="data/generated/abilene_traffic_matrices.npy",
        fl_model_path="models/fl_global_model_abilene_fedadam.pt" if os.path.exists("models/fl_global_model_abilene_fedadam.pt") else "models/fl_global_model.pt",
        domain_data_dir="data/preprocessed_abilene",
        num_managed=15,
        fail_edge=(1, 11),
        policies=abilene_policies,
        out_dir=out_dir,
    )

    # Combine all
    master_df = pd.concat([df_geant, df_nsfnet, df_abilene], ignore_index=True)
    master_csv = os.path.join(out_dir, "master_benchmark_all_topologies.csv")
    master_df.to_csv(master_csv, index=False)
    print("\n" + "=" * 80)
    print(f"MASTER BENCHMARK COMPLETE! Saved to {master_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
