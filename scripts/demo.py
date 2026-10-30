"""
Demo script for FL+RL Traffic Engineering prototype.
Compares Trained FL+RL Agent against Untrained Baselines (OSPF Shortest-Path & Random Agent)
across ALL test scenarios (T1 Normal, T2 Spike, T3 Gradual, T4 Flash Crowd, T5 Link Failure)
over multiple random seeds/tries with statistical reporting (mean ± std).

Produces rich multi-metric comparative plots in results/demo/:
    - overview_all_scenarios_mlu.png: 5-panel MLU timeline
    - overview_all_scenarios_delay.png: 5-panel End-to-End Delay timeline
    - overview_all_scenarios_loss.png: 5-panel Packet Loss timeline
    - overview_all_scenarios_reward.png: 5-panel Step Reward timeline
    - scenario_t4_flash_crowd.png: 4-subplot deep dive into T4 (MLU, Loss, Delay, Reward)
    - scenario_t2_traffic_spike.png: 4-subplot deep dive into T2 (MLU, Loss, Delay, Reward)
    - summary_metrics_barchart.png: Grouped bar charts with error bars across all scenarios
"""

import argparse
import os
import sys
from pathlib import Path

# Add prototype root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def run_episode(env, policy_fn, start_idx: int, seed: int, scenario: str = "T1", num_steps: int = 100):
    """Run an episode using the given policy function under a specific scenario and seed."""
    env.reset(seed=seed)
    if hasattr(policy_fn, "reset"):
        policy_fn.reset()
    env.time_index = start_idx
    obs = env._get_observation()

    # Create scenario-specific copy of the TM slice
    original_slice = env.tm_data[start_idx : start_idx + num_steps].copy()
    scenario_slice = original_slice.copy()

    t2_s, t2_e = int(num_steps * 0.3), min(int(num_steps * 0.4), num_steps)
    t4_s, t4_e = int(num_steps * 0.5), min(int(num_steps * 0.55), num_steps)
    t6_m, t6_a = int(num_steps * 0.4), int(num_steps * 0.6)

    if scenario == "T2":
        for t in range(t2_s, t2_e):
            for k in range(min(3, len(env.managed_od_pairs))):
                u, v = env.managed_od_pairs[k]
                scenario_slice[t, u, v] *= 3.0
    elif scenario == "T3":
        for t in range(num_steps):
            scenario_slice[t] *= 1.0 + (0.5 * (t / float(num_steps)))
    elif scenario == "T4":
        for t in range(t4_s, t4_e):
            for k in range(min(5, len(env.managed_od_pairs))):
                u, v = env.managed_od_pairs[k]
                scenario_slice[t, u, v] *= 5.0
    elif scenario == "T6":
        for t in range(0, min(t6_m, num_steps)):
            scenario_slice[t] *= 2.0
        for t in range(min(t6_a, num_steps), num_steps):
            scenario_slice[t] *= 0.3

    env.tm_data[start_idx : start_idx + num_steps] = scenario_slice

    rewards = []
    mlus = []
    mean_utils = []
    avg_delays = []
    avg_losses = []
    churns = []

    # Determine dynamic link failure pair for T5 based on topology
    fail_link = None
    if scenario == "T5":
        if len(env.topology.nodes) == 22:
            fail_link = (4, 7)  # Frankfurt - Paris core link in GEANT
        elif len(env.topology.nodes) == 12:
            fail_link = (1, 11)  # Atlanta - Washington core link in Abilene
        elif env.topology.has_edge(8, 12):
            fail_link = (8, 12)  # NSFNET core link
        elif len(env.edges) > 0:
            fail_link = env.edges[0]

    t5_fail_step = int(num_steps * 0.4)
    for step in range(num_steps):
        # Apply link failure for T5 at step t5_fail_step
        if scenario == "T5" and step == t5_fail_step and fail_link:
            env.disable_link(fail_link[0], fail_link[1])

        action = policy_fn(obs, env)
        obs, reward, done, _truncated, info = env.step(action)

        rewards.append(reward)
        mlus.append(info["mlu"])
        mean_utils.append(float(np.mean(env.link_utils)))
        avg_delays.append(info["avg_delay"])
        avg_losses.append(info["avg_loss"] * 100.0)  # Convert to percentage
        churns.append(info.get("churn", 0.0) * 100.0)  # Convert to percentage

        if done:
            break

    # Cleanup link failure if modified
    if scenario == "T5" and fail_link:
        env.enable_link(fail_link[0], fail_link[1])

    # Restore original traffic matrix
    env.tm_data[start_idx : start_idx + num_steps] = original_slice

    return {
        "timesteps": np.arange(len(rewards)),
        "rewards": np.array(rewards),
        "mlus": np.array(mlus),
        "mean_utils": np.array(mean_utils),
        "avg_delays": np.array(avg_delays),
        "avg_losses": np.array(avg_losses),
        "churns": np.array(churns),
    }


def make_transfer_policy(agent, src_edges: int, src_managed: int, with_ospf_fallback: bool = True):
    """Zero-shot transfer adapter for cross-topology evaluation with congestion-guarded fallback."""
    is_rec = hasattr(agent, "policy") and hasattr(agent.policy, "lstm_actor")
    lstm_state = [None]
    ep_start = [True]
    current_action = [None]
    hold_count = [None]

    def reset_transfer():
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
        # 1. Current link utilizations
        padded[: min(n_e, src_edges)] = obs[: min(n_e, src_edges)]
        # 2. Predicted link utilizations
        padded[src_edges : src_edges + min(n_e, src_edges)] = obs[n_e : n_e + min(n_e, src_edges)]
        # 3. Demands
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

        if not with_ospf_fallback:
            return raw_action

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

            thresh = 0.65 if len(env.topology.nodes) == 14 else 0.70
            return_thresh = thresh - 0.05

            if curr_p == 0:
                # If path 0 is congested (>= thresh per topology) or broken, look for candidate
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
                # Flow is on a detour. Check if candidate path failed
                curr_path = paths[curr_p]
                curr_edges = [(curr_path[i], curr_path[i + 1]) for i in range(len(curr_path) - 1)]
                curr_has_fail = any(e in disabled for e in curr_edges)

                # Projected load on Path 0 if flow returns (load = background + demand / capacity)
                p0_caps = [env.topology[e[0]][e[1]]["capacity"] for e in p0_edges]
                projected_p0 = max([u_val + (demand / c) for u_val, c in zip(p0_utils, p0_caps)]) if p0_utils else 0.0

                if curr_has_fail:
                    current_action[0][k] = 0
                    hold_count[0][k] = 0
                elif hold_count[0][k] >= 15:  # Hold for at least 15 steps during traffic surge
                    # Only return if Path 0 can absorb the flow WITHOUT re-congesting!
                    if projected_p0 < return_thresh and not p0_has_fail:
                        current_action[0][k] = 0
                        hold_count[0][k] = 0

        return current_action[0].copy()

    policy_fn.reset = reset_transfer
    return policy_fn


def run_demo_for_topology(topology: str, args, base_results_dir: Path):
    if topology == "nsfnet":
        topo_path = "data/nsfnet_topology.json"
        tm_path = "data/generated/traffic_matrices.npy"
        domain_data_dir = "data/preprocessed"
        num_managed_od = 20
        fl_model_path = "models/fl_global_model.pt"
        rl_model_path = "models/ppo_agent.zip"
        results_dir = base_results_dir / "nsfnet"
    elif topology == "geant":
        topo_path = "data/geant_topology.json"
        tm_path = "data/generated/geant_traffic_matrices.npy"
        domain_data_dir = "data/preprocessed_geant"
        num_managed_od = 30
        fl_model_path = (
            "models/fl_global_model_geant_fedadam.pt"
            if os.path.exists("models/fl_global_model_geant_fedadam.pt")
            else "models/fl_global_model.pt"
        )
        rl_model_path = (
            "models/ppo_agent_geant.zip" if os.path.exists("models/ppo_agent_geant.zip") else "models/ppo_agent.zip"
        )
        results_dir = base_results_dir / "geant"
    else:  # abilene
        topo_path = "data/abilene_topology.json"
        tm_path = "data/generated/abilene_traffic_matrices.npy"
        domain_data_dir = "data/preprocessed_abilene"
        num_managed_od = 15
        fl_model_path = (
            "models/fl_global_model_abilene_fedadam.pt"
            if os.path.exists("models/fl_global_model_abilene_fedadam.pt")
            else "models/fl_global_model.pt"
        )
        if os.path.exists("models/ppo_agent_abilene.zip") and not getattr(args, "cross_eval", False):
            rl_model_path = "models/ppo_agent_abilene.zip"
        elif os.path.exists("models/ppo_agent_geant.zip"):
            rl_model_path = "models/ppo_agent_geant.zip"
        else:
            rl_model_path = "models/ppo_agent.zip"
        results_dir = base_results_dir / "abilene"

    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK DEMO FOR TOPOLOGY: {topology.upper()}")
    print(f"Topology JSON: {topo_path} | TM: {tm_path}")
    print(f"Results Directory: {results_dir}")
    print(f"{'=' * 80}")

    if not (os.path.exists(topo_path) and os.path.exists(tm_path)):
        print(f"[{topology.upper()}] Data files not found ({topo_path} or {tm_path}).")
        print(f"Generating on the fly for {topology}...")
        from data.generate_tm import build_graph, generate_base_tms, generate_temporal_tms, load_topology, scale_alpha

        topo_data = load_topology(topo_path)
        G = build_graph(topo_data)
        base_tm = generate_base_tms(G)
        alpha = scale_alpha(G, base_tm, target_utilization=0.85)
        tms = generate_temporal_tms(base_tm, alpha, num_timesteps=max(args.steps * 10, 1000))
        os.makedirs(os.path.dirname(tm_path), exist_ok=True)
        np.save(tm_path, tms)

    from rl.env import NetworkRoutingEnv

    env = NetworkRoutingEnv(
        topo_path=topo_path,
        tm_path=tm_path,
        fl_model_path=fl_model_path if os.path.exists(fl_model_path) else None,
        domain_data_dir=domain_data_dir,
        num_managed_od=num_managed_od,
    )

    trained_agent = None
    is_cross_eval = False
    is_recurrent = False
    custom_objects = {
        "lr_schedule": lambda _: 0.0,
        "learning_rate": 0.0,
    }
    if os.path.exists(rl_model_path):
        try:
            from sb3_contrib import RecurrentPPO

            trained_agent = RecurrentPPO.load(rl_model_path, custom_objects=custom_objects)
            is_recurrent = True
            print(f"[{topology.upper()}] Loaded RecurrentPPO (MlpLstmPolicy) agent from {rl_model_path}.")
        except Exception:
            try:
                from stable_baselines3 import PPO

                trained_agent = PPO.load(rl_model_path, custom_objects=custom_objects)
                is_recurrent = False
                print(f"[{topology.upper()}] Loaded standard PPO agent from {rl_model_path}.")
            except Exception as e:
                print(f"[{topology.upper()}] Warning: Failed to load RL agent ({e}).")

        if (
            trained_agent is not None
            and topology == "abilene"
            and ("geant" in rl_model_path or "ppo_agent.zip" in rl_model_path)
        ):
            is_cross_eval = True
            print(f"[{topology.upper()}] Running CROSS-TOPOLOGY transfer evaluation (Source: {rl_model_path})!")
    else:
        print(f"[{topology.upper()}] RL agent {rl_model_path} not found. Running OSPF & Random baselines.")

    # Policies
    def policy_ospf(_obs, env):
        return np.zeros(env.num_managed, dtype=np.int32)

    def policy_random(_obs, env):
        return env.action_space.sample()

    if trained_agent is not None:
        if is_cross_eval:
            src_edges = 72 if "geant" in rl_model_path else 44
            src_managed = 30 if "geant" in rl_model_path else 20
            policy_trained = make_transfer_policy(trained_agent, src_edges, src_managed)
        else:
            lstm_state = [None]
            ep_start = [True]

            def reset_trained():
                lstm_state[0] = None
                ep_start[0] = True

            def policy_trained(obs, env):
                try:
                    if is_recurrent:
                        raw_action, lstm_state[0] = trained_agent.predict(
                            obs,
                            state=lstm_state[0],
                            episode_start=np.array([ep_start[0]]),
                            deterministic=True,
                        )
                        ep_start[0] = False
                        return raw_action[: env.num_managed]

                    raw_action, _ = trained_agent.predict(obs, deterministic=True)
                    return raw_action[: env.num_managed]
                except Exception:
                    return np.zeros(env.num_managed, dtype=np.int32)

            policy_trained.reset = reset_trained
    else:

        def policy_trained(_obs, env):
            return np.zeros(env.num_managed, dtype=np.int32)

    policies = {
        "OSPF": policy_ospf,
    }
    if not args.no_random:
        policies["Random"] = policy_random
    policies["Trained"] = policy_trained

    # If topology is abilene and geant model is available, include GEANT Transfer policy
    if (topology == "abilene" or (topology == "nsfnet" and getattr(args, "cross_eval", False))) and os.path.exists("models/ppo_agent_geant.zip"):
        try:
            try:
                from sb3_contrib import RecurrentPPO

                geant_model = RecurrentPPO.load("models/ppo_agent_geant.zip", custom_objects=custom_objects)
            except Exception:
                geant_model = PPO.load("models/ppo_agent_geant.zip", custom_objects=custom_objects)
            policies["GEANT Transfer"] = make_transfer_policy(
                geant_model, src_edges=72, src_managed=30, with_ospf_fallback=True
            )
            print(f"[{topology.upper()}] Added 'GEANT Transfer' policy (GÉANT PPO with OSPF fallback) to evaluation!")
        except Exception as e:
            print(f"[{topology.upper()}] Could not load GEANT model for transfer: {e}")

    # If a fine-tuned GEANT model exists for this topology, include it in evaluation
    finetuned_path = f"models/ppo_agent_geant_finetuned_{topology}.zip"
    if os.path.exists(finetuned_path):
        try:
            try:
                from sb3_contrib import RecurrentPPO

                finetuned_model = RecurrentPPO.load(finetuned_path, custom_objects=custom_objects)
            except Exception:
                finetuned_model = PPO.load(finetuned_path, custom_objects=custom_objects)
            policies["GEANT Finetuned"] = make_transfer_policy(
                finetuned_model, src_edges=72, src_managed=30, with_ospf_fallback=True
            )
            print(f"[{topology.upper()}] Added 'GEANT Finetuned' policy to evaluation!")
        except Exception as e:
            print(f"[{topology.upper()}] Could not load fine-tuned model: {e}")

    active_policy_names = list(policies.keys())

    scenarios = ["T1", "T2", "T3", "T4", "T5", "T6"]
    scenario_titles = {
        "T1": "T1: Normal Traffic",
        "T2": "T2: Traffic Spike (3x at t=30..40)",
        "T3": "T3: Gradual Load (+50% over 100 steps)",
        "T4": "T4: Flash Crowd (5x at t=50..55)",
        "T5": "T5: Link Failure",
        "T6": "T6: Irregular Day (Busy Morning, Quiet Afternoon)",
    }

    print(
        f"\nRunning benchmark for {topology.upper()} across {len(scenarios)} scenarios and {len(args.seeds)} random seeds: {args.seeds}..."
    )

    all_runs = {sc: {p: [] for p in policies} for sc in scenarios}

    for s_idx, seed in enumerate(args.seeds):
        day_idx = (seed * 3) % 20 + 1
        start_idx = min(day_idx * 288, max(0, env.dataset_len - args.steps - 1))

        for sc in scenarios:
            for p_name, p_fn in policies.items():
                run_data = run_episode(env, p_fn, start_idx, seed=seed, scenario=sc, num_steps=args.steps)
                all_runs[sc][p_name].append(run_data)

    policy_labels = {
        "OSPF": "OSPF (Shortest Path)",
        "Random": "Untrained (Random)",
        "Trained": "Trained FL+RL Agent",
        "GEANT Transfer": "GEANT Transferred (Fallback)",
        "GEANT Finetuned": "GEANT Finetuned (Few-shot)",
    }
    policy_plot_labels = {
        "OSPF": "OSPF-SPF",
        "Random": "Random Policy",
        "Trained": "Trained FL+RL",
        "GEANT Transfer": "GEANT Transfer",
        "GEANT Finetuned": "GEANT Finetuned",
    }

    def get_policy_label(p_name):
        return policy_labels.get(p_name, p_name)

    def get_policy_plot_label(p_name):
        return policy_plot_labels.get(p_name, p_name)

    # --- Print Statistical Summary Table (Mean ± Std) ---
    print("\n" + "=" * 122)
    print(
        f"{'SCENARIO':<6} | {'POLICY':<20} | {'MEAN MLU':<14} | {'MAX MLU':<14} | {'DELAY (ms)':<14} | {'LOSS (%)':<14} | {'CHURN (%)':<14} | {'REWARD':<14}"
    )
    print("=" * 122)

    summary_stats = {sc: {p: {} for p in policies} for sc in scenarios}

    for sc in scenarios:
        for p_name in active_policy_names:
            seed_runs = all_runs[sc][p_name]

            # Aggregate across seeds
            mean_mlus = [np.mean(r["mlus"]) for r in seed_runs]
            max_mlus = [np.max(r["mlus"]) for r in seed_runs]
            delays = [np.mean(r["avg_delays"]) for r in seed_runs]
            losses = [np.mean(r["avg_losses"]) for r in seed_runs]
            churns = [np.mean(r["churns"]) for r in seed_runs]
            rewards = [np.mean(r["rewards"]) for r in seed_runs]

            summary_stats[sc][p_name] = {
                "mlu_mean": np.mean(mean_mlus),
                "mlu_std": np.std(mean_mlus),
                "max_mlu_mean": np.mean(max_mlus),
                "max_mlu_std": np.std(max_mlus),
                "delay_mean": np.mean(delays),
                "delay_std": np.std(delays),
                "loss_mean": np.mean(losses),
                "loss_std": np.std(losses),
                "churn_mean": np.mean(churns),
                "churn_std": np.std(churns),
                "reward_mean": np.mean(rewards),
                "reward_std": np.std(rewards),
            }

            label = get_policy_label(p_name)

            st = summary_stats[sc][p_name]
            print(
                f"{sc:<6} | {label:<20} | "
                f"{st['mlu_mean']:.3f} ± {st['mlu_std']:.3f} | "
                f"{st['max_mlu_mean']:.3f} ± {st['max_mlu_std']:.3f} | "
                f"{st['delay_mean']:5.2f} ± {st['delay_std']:4.2f} | "
                f"{st['loss_mean']:5.2f} ± {st['loss_std']:4.2f} | "
                f"{st['churn_mean']:5.2f} ± {st['churn_std']:4.2f} | "
                f"{st['reward_mean']:6.2f} ± {st['reward_std']:4.2f}"
            )
        print("-" * 122)

    # --- Plot Generation ---
    print(f"\nGenerating rich comparative plots in {results_dir.resolve()}...")

    colors = {
        "OSPF": "#1f77b4",
        "Random": "#ff7f0e",
        "Trained": "#2ca02c",
        "GEANT Transfer": "#9467bd",
        "GEANT Finetuned": "#d62728",
    }
    styles = {
        "OSPF": "-",
        "Random": "--",
        "Trained": "-",
        "GEANT Transfer": "-.",
        "GEANT Finetuned": "--",
    }
    widths = {
        "OSPF": 1.8,
        "Random": 1.2,
        "Trained": 2.2,
        "GEANT Transfer": 2.0,
        "GEANT Finetuned": 2.2,
    }
    alphas = {
        "OSPF": 0.9,
        "Random": 0.6,
        "Trained": 1.0,
        "GEANT Transfer": 0.9,
        "GEANT Finetuned": 0.95,
    }

    # 1. Five-Panel Timelines for All Metrics (Averaged across seeds with std shading)
    metric_configs = [
        ("mlus", "Maximum Link Utilization (MLU)", "overview_all_scenarios_mlu.png", True),
        ("avg_delays", "End-to-End Delay (ms)", "overview_all_scenarios_delay.png", False),
        ("avg_losses", "Packet Loss Ratio (%)", "overview_all_scenarios_loss.png", False),
        ("rewards", "Per-Step Reward", "overview_all_scenarios_reward.png", False),
    ]

    def setup_axis_ticks(ax, n_steps: int):
        ax.set_xlim(0, max(1, n_steps - 1))
        ticks = np.linspace(0, max(1, n_steps - 1), 6, dtype=int)
        labels = []
        for t in ticks:
            total_minutes = 8 * 60 + t * 5
            h = (total_minutes // 60) % 24
            m = total_minutes % 60
            labels.append(f"{h:02d}:{m:02d}")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)

    for metric_key, ylabel, fname, has_threshold in metric_configs:
        _fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
        for idx, sc in enumerate(scenarios):
            ax = axes[idx]
            timesteps = all_runs[sc]["OSPF"][0]["timesteps"]

            for p_name in active_policy_names:
                # Stack all seeds for this metric
                matrix = np.array([r[metric_key] for r in all_runs[sc][p_name]])  # (n_seeds, num_steps)
                mean_curve = np.mean(matrix, axis=0)
                std_curve = np.std(matrix, axis=0)

                label_p = get_policy_plot_label(p_name)
                ax.plot(
                    timesteps,
                    mean_curve,
                    label=label_p,
                    color=colors.get(p_name, "#9467bd"),
                    linestyle=styles.get(p_name, "-"),
                    linewidth=widths.get(p_name, 1.8),
                    alpha=alphas.get(p_name, 0.9),
                )
                ax.fill_between(
                    timesteps,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=colors.get(p_name, "#9467bd"),
                    alpha=0.15,
                )

            if has_threshold:
                ax.axhline(
                    0.90, color="red", linestyle=":", alpha=0.7, label="Congestion Threshold (0.90)" if idx == 0 else ""
                )

            ax.set_title(scenario_titles[sc], fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=10)
            setup_axis_ticks(ax, len(timesteps))
            ax.grid(True, linestyle="--", alpha=0.5)
            if idx == 0:
                ax.legend(loc="upper right", framealpha=0.9)

        axes[-1].set_xlabel("Time of Day (08:00 AM - 04:20 PM / 5-minute intervals)", fontsize=11)
        plt.tight_layout()
        plt.savefig(results_dir / fname, dpi=200)
        plt.savefig(results_dir / fname.replace(".png", ".pdf"), bbox_inches="tight")
        plt.savefig(results_dir / fname.replace(".png", ".svg"), bbox_inches="tight")
        plt.close()

    # 2. Comprehensive 4-Subplot Deep Dive into Scenario T4 (Flash Crowd)
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t4_metrics = [
        ("mlus", "MLU", axes[0, 0], True),
        ("avg_losses", "Packet Loss (%)", axes[0, 1], False),
        ("avg_delays", "End-to-End Delay (ms)", axes[1, 0], False),
        ("rewards", "Reward", axes[1, 1], False),
    ]

    timesteps = all_runs["T4"]["OSPF"][0]["timesteps"]
    for m_key, m_label, ax, thresh in t4_metrics:
        for p_name in active_policy_names:
            matrix = np.array([r[m_key] for r in all_runs["T4"][p_name]])
            mean_c = np.mean(matrix, axis=0)
            std_c = np.std(matrix, axis=0)

            p_lbl = get_policy_plot_label(p_name)
            ax.plot(
                timesteps,
                mean_c,
                label=p_lbl,
                color=colors.get(p_name, "#9467bd"),
                linewidth=widths.get(p_name, 1.8),
                linestyle=styles.get(p_name, "-"),
            )
            ax.fill_between(timesteps, mean_c - std_c, mean_c + std_c, color=colors.get(p_name, "#9467bd"), alpha=0.15)

        t4_s = int(len(timesteps) * 0.5)
        t4_e = min(int(len(timesteps) * 0.55), len(timesteps))
        ax.axvspan(t4_s, t4_e, color="purple", alpha=0.18, label=f"Flash Crowd Surge (5x, t={t4_s}..{t4_e})")
        if thresh:
            ax.axhline(0.90, color="red", linestyle=":", label="Threshold (0.90)")
        ax.set_title(f"T4 Flash Crowd: {m_label}", fontsize=11, fontweight="bold")
        setup_axis_ticks(ax, len(timesteps))
        ax.set_xlabel("Time of Day (08:00 AM - 04:20 PM / 5-min intervals)")
        ax.set_ylabel(m_label)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.suptitle(
        "Scenario T4 (Flash Crowd Surge) — Comprehensive Multi-Metric Comparison", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(results_dir / "scenario_t4_flash_crowd.png", dpi=200)
    plt.savefig(results_dir / "scenario_t4_flash_crowd.pdf", bbox_inches="tight")
    plt.savefig(results_dir / "scenario_t4_flash_crowd.svg", bbox_inches="tight")
    plt.close()

    # 3. Comprehensive 4-Subplot Deep Dive into Scenario T2 (Traffic Spike)
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t2_metrics = [
        ("mlus", "MLU", axes[0, 0], True),
        ("avg_losses", "Packet Loss (%)", axes[0, 1], False),
        ("avg_delays", "End-to-End Delay (ms)", axes[1, 0], False),
        ("rewards", "Reward", axes[1, 1], False),
    ]

    timesteps = all_runs["T2"]["OSPF"][0]["timesteps"]
    for m_key, m_label, ax, thresh in t2_metrics:
        for p_name in active_policy_names:
            matrix = np.array([r[m_key] for r in all_runs["T2"][p_name]])
            mean_c = np.mean(matrix, axis=0)
            std_c = np.std(matrix, axis=0)

            p_lbl = get_policy_plot_label(p_name)
            ax.plot(
                timesteps,
                mean_c,
                label=p_lbl,
                color=colors.get(p_name, "#9467bd"),
                linewidth=widths.get(p_name, 1.8),
                linestyle=styles.get(p_name, "-"),
            )
            ax.fill_between(timesteps, mean_c - std_c, mean_c + std_c, color=colors.get(p_name, "#9467bd"), alpha=0.15)

        t2_s = int(len(timesteps) * 0.3)
        t2_e = min(int(len(timesteps) * 0.4), len(timesteps))
        ax.axvspan(t2_s, t2_e, color="purple", alpha=0.18, label=f"Spike Surge (3x, t={t2_s}..{t2_e})")
        if thresh:
            ax.axhline(0.90, color="red", linestyle=":", label="Threshold (0.90)")
        ax.set_title(f"T2 Traffic Spike: {m_label}", fontsize=11, fontweight="bold")
        setup_axis_ticks(ax, len(timesteps))
        ax.set_xlabel("Time of Day (08:00 AM - 04:20 PM / 5-min intervals)")
        ax.set_ylabel(m_label)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.suptitle(
        "Scenario T2 (Traffic Spike 3x) — Comprehensive Multi-Metric Comparison", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(results_dir / "scenario_t2_traffic_spike.png", dpi=200)
    plt.savefig(results_dir / "scenario_t2_traffic_spike.pdf", bbox_inches="tight")
    plt.savefig(results_dir / "scenario_t2_traffic_spike.svg", bbox_inches="tight")
    plt.close()

    # 4. Comprehensive 4-Subplot Deep Dive into Scenario T5 (Link Failure)
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t5_metrics = [
        ("mlus", "MLU", axes[0, 0], True),
        ("avg_losses", "Packet Loss (%)", axes[0, 1], False),
        ("avg_delays", "End-to-End Delay (ms)", axes[1, 0], False),
        ("rewards", "Reward", axes[1, 1], False),
    ]

    timesteps = all_runs["T5"]["OSPF"][0]["timesteps"]
    for m_key, m_label, ax, thresh in t5_metrics:
        for p_name in active_policy_names:
            matrix = np.array([r[m_key] for r in all_runs["T5"][p_name]])
            mean_c = np.mean(matrix, axis=0)
            std_c = np.std(matrix, axis=0)

            p_lbl = get_policy_plot_label(p_name)
            ax.plot(
                timesteps,
                mean_c,
                label=p_lbl,
                color=colors.get(p_name, "#9467bd"),
                linewidth=widths.get(p_name, 1.8),
                linestyle=styles.get(p_name, "-"),
            )
            ax.fill_between(timesteps, mean_c - std_c, mean_c + std_c, color=colors.get(p_name, "#9467bd"), alpha=0.15)

        t5_f = int(len(timesteps) * 0.4)
        ax.axvline(t5_f, color="darkred", linestyle="--", linewidth=1.8, label=f"Link Fails at t={t5_f}")
        if thresh:
            ax.axhline(0.90, color="red", linestyle=":", label="Threshold (0.90)")
        ax.set_title(f"T5 Link Failure: {m_label}", fontsize=11, fontweight="bold")
        setup_axis_ticks(ax, len(timesteps))
        ax.set_xlabel("Time of Day (08:00 AM - 04:20 PM / 5-min intervals)")
        ax.set_ylabel(m_label)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.suptitle(
        "Scenario T5 (Backbone Link Failure) — Comprehensive Multi-Metric Comparison", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(results_dir / "scenario_t5_link_failure.png", dpi=200)
    plt.savefig(results_dir / "scenario_t5_link_failure.pdf", bbox_inches="tight")
    plt.savefig(results_dir / "scenario_t5_link_failure.svg", bbox_inches="tight")
    plt.close()

    # 4b. Comprehensive 4-Subplot Deep Dive into Scenario T6 (Irregular Day)
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t6_metrics = [
        ("mlus", "MLU", axes[0, 0], True),
        ("avg_losses", "Packet Loss (%)", axes[0, 1], False),
        ("avg_delays", "End-to-End Delay (ms)", axes[1, 0], False),
        ("rewards", "Reward", axes[1, 1], False),
    ]

    timesteps = all_runs["T6"]["OSPF"][0]["timesteps"]
    for m_key, m_label, ax, thresh in t6_metrics:
        for p_name in active_policy_names:
            matrix = np.array([r[m_key] for r in all_runs["T6"][p_name]])
            mean_c = np.mean(matrix, axis=0)
            std_c = np.std(matrix, axis=0)

            p_lbl = get_policy_plot_label(p_name)
            ax.plot(
                timesteps,
                mean_c,
                label=p_lbl,
                color=colors.get(p_name, "#9467bd"),
                linewidth=widths.get(p_name, 1.8),
                linestyle=styles.get(p_name, "-"),
            )
            ax.fill_between(timesteps, mean_c - std_c, mean_c + std_c, color=colors.get(p_name, "#9467bd"), alpha=0.15)

        t6_m = int(len(timesteps) * 0.4)
        t6_a = int(len(timesteps) * 0.6)
        ax.axvspan(0, t6_m, color="orange", alpha=0.15, label=f"Busy Morning (2x, t=0..{t6_m})")
        ax.axvspan(
            t6_a,
            len(timesteps) - 1,
            color="green",
            alpha=0.15,
            label=f"Quiet Afternoon (0.3x, t={t6_a}..{len(timesteps) - 1})",
        )
        if thresh:
            ax.axhline(0.90, color="red", linestyle=":", label="Threshold (0.90)")
        ax.set_title(f"T6 Irregular Day: {m_label}", fontsize=11, fontweight="bold")
        setup_axis_ticks(ax, len(timesteps))
        ax.set_xlabel("Time of Day (08:00 AM - 04:20 PM / 5-min intervals)")
        ax.set_ylabel(m_label)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.suptitle(
        "Scenario T6 (Irregular Day: Busy Morning, Quiet Afternoon) — Multi-Metric Comparison",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(results_dir / "scenario_t6_irregular_day.png", dpi=200)
    plt.savefig(results_dir / "scenario_t6_irregular_day.pdf", bbox_inches="tight")
    plt.savefig(results_dir / "scenario_t6_irregular_day.svg", bbox_inches="tight")
    plt.close()

    # 5. Summary Grouped Bar Chart Across Scenarios (with Error Bars)
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bar_metrics = [
        ("mlu_mean", "mlu_std", "Mean MLU (Lower is better)", axes[0, 0]),
        ("delay_mean", "delay_std", "Mean Delay (ms) (Lower is better)", axes[0, 1]),
        ("loss_mean", "loss_std", "Packet Loss (%) (Lower is better)", axes[1, 0]),
        ("reward_mean", "reward_std", "Mean Reward (Higher is better)", axes[1, 1]),
    ]

    x = np.arange(len(scenarios))
    num_p = len(active_policy_names)
    bar_width = 0.8 / max(1, num_p)

    for m_mean, m_std, title, ax in bar_metrics:
        for i, p_name in enumerate(active_policy_names):
            vals = [summary_stats[sc][p_name][m_mean] for sc in scenarios]
            errs = [summary_stats[sc][p_name][m_std] for sc in scenarios]

            p_lbl = get_policy_plot_label(p_name)
            offset = (i - (num_p - 1) / 2.0) * bar_width
            ax.bar(
                x + offset,
                vals,
                bar_width,
                yerr=errs,
                capsize=4,
                label=p_lbl,
                color=colors.get(p_name, "#9467bd"),
                alpha=0.85,
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax.legend(loc="upper right" if "Reward" not in title else "lower right", framealpha=0.9, fontsize=9)

    plt.suptitle(
        "Overall Performance Benchmark Across All Scenarios (5-Seed Mean ± Std)", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(results_dir / "summary_metrics_barchart.png", dpi=200)
    plt.savefig(results_dir / "summary_metrics_barchart.pdf", bbox_inches="tight")
    plt.savefig(results_dir / "summary_metrics_barchart.svg", bbox_inches="tight")
    plt.close()

    print(f"[{topology.upper()}] All multi-metric comparative plots successfully generated in {results_dir}!")

    if topology == "nsfnet" and results_dir != base_results_dir:
        import shutil

        for img in results_dir.glob("*.png"):
            shutil.copy2(img, base_results_dir / img.name)
        print(f"[{topology.upper()}] Mirrored plots to root {base_results_dir} for dual accessibility.")


def run_demo():
    parser = argparse.ArgumentParser(description="Multi-Scenario & Multi-Seed FL+RL Evaluation Demo across Topologies")
    parser.add_argument(
        "--topology",
        type=str,
        default="all",
        choices=["all", "nsfnet", "geant", "abilene"],
        help="Topology to evaluate: nsfnet, geant, abilene, or all",
    )
    parser.add_argument("--cross_eval", action="store_true", help="Explicitly test GEANT-trained model across other topologies (Abilene/NSFNET)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds for testing")
    parser.add_argument("--steps", type=int, default=100, help="Timesteps per episode")
    parser.add_argument("--out_dir", type=str, default="results/demo", help="Base directory to save generated plots")
    parser.add_argument("--no_random", action="store_true", help="Skip the Untrained (Random Policy) baseline")
    args = parser.parse_args()

    base_results_dir = Path(args.out_dir)
    topologies = ["nsfnet", "geant", "abilene"] if args.topology == "all" else [args.topology]

    print(f"Starting Multi-Model Evaluation Demo for topologies: {topologies}")
    for topo in topologies:
        run_demo_for_topology(topo, args, base_results_dir)

    print(f"\nAll benchmark demo evaluations for {topologies} successfully finished!")


if __name__ == "__main__":
    run_demo()
