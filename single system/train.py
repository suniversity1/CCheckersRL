import os
import time
import random

import numpy as np
from gymnasium import Env
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule
import torch as th

from pettingzoo_checkers import CheckersParallelEnv

CheckersEnvironment = CheckersParallelEnv
OBS_MODE = "mlp"
POLICY_TYPE = "MlpPolicy"
CHECKPOINT_DIR_NAME = "checkpoints_mlp"


# Import for Metrics 
import csv
from stable_baselines3.common.callbacks import BaseCallback


HALF_COLOURS_1 = ["red", "lawn green", "yellow"]
HALF_COLOURS_2 = ["blue", "gray0", "purple"]

ALL_COLOURS = HALF_COLOURS_1 + HALF_COLOURS_2

PHASES = [
    ("red", "blue"),
    ("blue", "red"),
    ("lawn green", "gray0"),
    ("gray0", "lawn green"),
    ("yellow", "purple"),
    ("purple", "yellow"),
]


def hex_distance(q1, r1, q2, r2):
    s1 = -q1 - r1
    s2 = -q2 - r2
    return max(abs(q1 - q2), abs(r1 - r2), abs(s1 - s2))


def evaluate_move_heuristic(env, colour, action_id, noise=0.0):
    pin_id, dest = env.decode_action(action_id)
    pin = next(p for p in env.boardPins if p.color == colour and p.id == pin_id)

    board = env.board
    old_cell = board.cells[pin.axialindex]
    new_cell = board.cells[dest]

    target_colour = board.colour_opposites[colour]
    targets = board.axial_of_colour(target_colour)

    old_dist = min(hex_distance(old_cell.q, old_cell.r, board.cells[t].q, board.cells[t].r) for t in targets)
    new_dist = min(hex_distance(new_cell.q, new_cell.r, board.cells[t].q, board.cells[t].r) for t in targets)

    score = old_dist - new_dist
    move_dist = hex_distance(old_cell.q, old_cell.r, new_cell.q, new_cell.r)
    if move_dist >= 2:
        score += 0.25 * move_dist
    if dest in set(targets):
        score += 0.5
    if noise > 0:
        score += np.random.normal(0, noise)
    return score


def heuristic_action(env, colour=None, opponent_type="medium"):
    colour = colour or env.current_agent_color

    legal = env.legal_actions(colour)

    if not legal:
        return None

    if opponent_type == "weak":
        epsilon = 0.55
        top_k = 10
        noise = 0.18

    elif opponent_type == "medium":
        epsilon = 0.35
        top_k = 6
        noise = 0.10

    else:  # strong
        epsilon = 0.10
        top_k = 3
        noise = 0.00

    if np.random.rand() < epsilon:
        return int(np.random.choice(legal))

    scored = [
        (
            evaluate_move_heuristic(
                env,
                colour,
                action,
                noise=noise
            ),
            action
        )
        for action in legal
    ]

    scored.sort(reverse=True, key=lambda item: item[0])

    k = min(top_k, len(scored))

    return int(
        np.random.choice(
            [action for _, action in scored[:k]]
        )
    )


class ModelVsHeuristicEnv(Env):
    def __init__(self, model_colour, opponent_colour, max_turns_per_player=200, recent_positions_per_pin=3,min_opponents=1,max_opponents=3, loaded_checkpoints=None,):
        super().__init__()
        self.model_colour = model_colour
        self.opponent_colour = opponent_colour
        self.max_turns_per_player = max_turns_per_player
        self.recent_positions_per_pin = recent_positions_per_pin
        self.min_opponents = min_opponents
        self.max_opponents = max_opponents
        self.env = None
        self.observation_space = None
        self.action_space = None
        self.current_agent_color = None
        self.loaded_checkpoints = loaded_checkpoints or []

        self._build_env()

    def _build_env(self):
        # Always include the opposite/main opponent
        chosen_colours = [self.model_colour, self.opponent_colour]
        self.opponent_strengths = {}

        possible_extra_opponents = [
            c for c in ALL_COLOURS
            if c not in chosen_colours
        ]

        num_opponents = np.random.randint(
            self.min_opponents,
            self.max_opponents + 1
        )

        num_extra = max(0, num_opponents - 1)
        num_extra = min(num_extra, len(possible_extra_opponents))

        dynamic_max_turns = (
            self.max_turns_per_player
            + 20 * (num_opponents - 1)
        )

        if num_extra > 0:
            extra_opponents = list(
                np.random.choice(
                    possible_extra_opponents,
                    size=num_extra,
                    replace=False
                )
            )
            chosen_colours.extend(extra_opponents)

        for colour in chosen_colours:
            if colour == self.model_colour:
                continue

            # Only the main opponent can become a checkpoint opponent
            if (
                colour == self.opponent_colour
                and len(self.loaded_checkpoints) > 0
            ):
                self.opponent_strengths[colour] = np.random.choice(
                    ["medium", "strong", "checkpoint"],
                    p=[0.20, 0.30, 0.50]
                )

            else:
                self.opponent_strengths[colour] = np.random.choice(
                    ["weak", "medium", "strong"],
                    p=[0.25, 0.50, 0.25]
                )

        self.env = CheckersEnvironment(
            num_players=len(chosen_colours),
            max_turns_per_player=dynamic_max_turns,
            recent_positions_per_pin=self.recent_positions_per_pin,
            player_colours=chosen_colours,
        )

        self.observation_space = self.env.observation_spaces[self.env.agents[0]]
        self.action_space = self.env.action_spaces[self.env.agents[0]]
        self.current_agent_color = self.env.agent_colours[self.env.current_agent]

    def reset(self, seed=None, options=None):
        self._build_env()

        self.env.reset(seed=seed, options=options)
        self.current_agent_color = self.env.agent_colours[self.env.current_agent]

        obs = self.env._observe(
            self._agent_name_for_colour(self.model_colour)
        )

        return obs, {}
    

    def _agent_name_for_colour(self, colour):
        return next(agent for agent, agent_colour in self.env.agent_colours.items() if agent_colour == colour)

    def action_masks(self):
        return self.env.action_mask(self.model_colour, normalized=True)

    def _step_heuristic_until_model_turn(self):
        total_reward = 0.0
        terminated = False
        truncated = False
        last_info = {}

        while not (terminated or truncated) and self.env.agent_colours[self.env.current_agent] != self.model_colour:
            opp_agent = self.env.current_agent
            opp_colour = self.env.agent_colours[opp_agent]

            opp_type = self.opponent_strengths[opp_colour]

            if opp_type == "checkpoint":
                if random.random() < 0.5:

                    checkpoint_model, checkpoint_vecnorm = random.choice(self.loaded_checkpoints)

                    opp_obs = self.env._observe(opp_agent)
                    
                    if checkpoint_vecnorm is not None:
                        opp_obs = checkpoint_vecnorm.normalize_obs(opp_obs)

                    mask = self.env.action_mask(
                        opp_colour,
                        normalized=True
                    )

                    if mask.sum() == 0:
                        truncated = True
                        break

                    opp_action, _ = checkpoint_model.predict(
                        opp_obs,
                        deterministic=True,
                        action_masks=mask,
                    )

                else:
                    opp_action = heuristic_action(
                        self.env,
                        colour=opp_colour,
                        opponent_type="strong",
                    )

            else:
                opp_action = heuristic_action(
                    self.env,
                    colour=opp_colour,
                    opponent_type=opp_type,
                )

            if opp_action is None:
                truncated = True
                break

            _, rewards, terminations, truncations, infos = self.env.step({opp_agent: opp_action})

            model_agent = self._agent_name_for_colour(self.model_colour)
            total_reward += rewards.get(model_agent, 0.0)

            terminated = any(terminations.values())
            truncated = any(truncations.values())
            last_info = infos.get(model_agent, {}) or infos.get(opp_agent, {})

        return total_reward, terminated, truncated, last_info
    
    def step(self, action):

        model_agent = self._agent_name_for_colour(self.model_colour)

        mask = self.env.action_mask(
            self.model_colour,
            normalized=True
        )

        if mask.sum() == 0:
            obs = self.env._observe(model_agent)

            return (
                obs,
                0.0,
                False,   # terminated
                True,    # truncated
                {"reason": "no_legal_actions"}
            )

        _, rewards, terminations, truncations, infos = self.env.step_normalized({model_agent: int(action)})

        total_reward = rewards.get(model_agent, 0.0)
        terminated = any(terminations.values())
        truncated = any(truncations.values())
        info = infos.get(model_agent, {})

        if not (terminated or truncated) and self.env.agent_colours[self.env.current_agent] != self.model_colour:
            extra_reward, opp_terminated, opp_truncated, opp_info = self._step_heuristic_until_model_turn()
            total_reward += extra_reward
            terminated = terminated or opp_terminated
            truncated = truncated or opp_truncated
            if opp_info:
                info.update(opp_info)

        obs = self.env._observe(model_agent)
        return obs, total_reward, terminated, truncated, info



def make_env(model_colour, opponent_colour, max_turns_per_player, recent_positions_per_pin, min_opponents=1, max_opponents=3, loaded_checkpoints=None):
    def _init():
        env = ModelVsHeuristicEnv(
            model_colour=model_colour,
            opponent_colour=opponent_colour,
            max_turns_per_player=max_turns_per_player,
            recent_positions_per_pin=recent_positions_per_pin,
            min_opponents=min_opponents,
            max_opponents=max_opponents,
            loaded_checkpoints=loaded_checkpoints,
        )
        return Monitor(env)

    return _init


class MetricsLoggerCallback(BaseCallback):
    def __init__(self, csv_path="training_metrics.csv", verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.csv_file = None
        self.writer = None
        self._recent_episode_rewards = []
        self._recent_episode_lengths = []

    def _on_training_start(self):
        self.csv_file = open(self.csv_path, "a", newline="")
        self.writer = csv.writer(self.csv_file)

        if self.csv_file.tell() == 0:
            self.writer.writerow([
                "timesteps",
                "ep_rew_mean",
                "ep_len_mean",
                "fps",
                "approx_kl",
                "clip_fraction",
                "entropy_loss",
                "explained_variance",
                "value_loss",
                "policy_gradient_loss",
                "learning_rate",
                "ent_coef",
                "vf_coef",
            ])

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict):
                episode = info.get("episode")
                if episode:
                    self._recent_episode_rewards.append(episode.get("r"))
                    self._recent_episode_lengths.append(episode.get("l"))
        return True

    def _on_rollout_end(self):
        logs = self.model.logger.name_to_value

        if self._recent_episode_rewards:
            ep_rew_mean = float(np.mean(self._recent_episode_rewards))
            ep_len_mean = float(np.mean(self._recent_episode_lengths))
        else:
            ep_rew_mean = logs.get("rollout/ep_rew_mean")
            ep_len_mean = logs.get("rollout/ep_len_mean")

        self.writer.writerow([
            self.num_timesteps,
            ep_rew_mean,
            ep_len_mean,
            logs.get("time/fps"),
            logs.get("train/approx_kl"),
            logs.get("train/clip_fraction"),
            logs.get("train/entropy_loss"),
            logs.get("train/explained_variance"),
            logs.get("train/value_loss"),
            logs.get("train/policy_gradient_loss"),
            logs.get("train/learning_rate"),
            getattr(self.model, "ent_coef", None),
            getattr(self.model, "vf_coef", None),
        ])
        self.csv_file.flush()
        self._recent_episode_rewards.clear()
        self._recent_episode_lengths.clear()

    def _on_training_end(self):
        if self.csv_file is not None:
            self.csv_file.close()


def evaluate_model(model, vec_env=None, num_games=20):
    wins = 0
    opponent_wins = 0
    truncations = 0

    total_reward = 0.0
    total_model_turns = 0
    total_pieces_in_target = 0
    total_pieces_in_start = 0

    for game_idx in range(num_games):
        print(f"Eval game {game_idx + 1}/{num_games}")

        model_colour = np.random.choice(ALL_COLOURS)

        opponent_colour = {
            "red": "blue",
            "lawn green": "gray0",
            "yellow": "purple",
            "blue": "red",
            "gray0": "lawn green",
            "purple": "yellow",
        }[model_colour]

        env = ModelVsHeuristicEnv(
            model_colour=model_colour,
            opponent_colour=opponent_colour,
            max_turns_per_player=100,
            recent_positions_per_pin=2,
            min_opponents=1,
            max_opponents=5,
        )

        obs, _ = env.reset()
        if vec_env is not None:
            obs = vec_env.normalize_obs(obs)

        done = False
        final_info = {}
        final_truncated = False

        model_agent = env._agent_name_for_colour(env.model_colour)

        start_time = time.time()

        while not done:
            mask = env.action_masks()

            if mask.sum() == 0:
                print("Model had no legal actions.")
                final_truncated  = True
                done = True
                break

            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=mask,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            if vec_env is not None:
                obs = vec_env.normalize_obs(obs)

            total_reward += reward

            done = terminated or truncated
            final_info = info
            final_truncated = truncated

            elapsed = time.time() - start_time
            model_turns = env.env.turns_per_player[model_agent]

            if elapsed > 60:
                print("\nBUG: evaluation timeout")
                print(f"elapsed={elapsed:.2f}s")
                print(f"model_colour={model_colour}")
                print(f"opponent_colour={opponent_colour}")
                print(f"players={len(env.env.agents)}")
                print(f"max_turns={env.env.max_turns_per_player}")
                print(f"model_turns={model_turns}")
                print(f"current_agent={env.env.current_agent}")
                print(f"current_colour={env.env.agent_colours[env.env.current_agent]}")
                env.env.render()
                final_truncated = True
                done = True
                break

            if model_turns > env.env.max_turns_per_player + 2:
                print("\nBUG: model turn limit exceeded")
                print(f"model_turns={model_turns}")
                print(f"max_turns={env.env.max_turns_per_player}")
                print(f"model_colour={model_colour}")
                print(f"opponent_colour={opponent_colour}")
                print(f"players={len(env.env.agents)}")
                print(f"current_agent={env.env.current_agent}")
                print(f"current_colour={env.env.agent_colours[env.env.current_agent]}")
                env.env.render()
                raise RuntimeError("Evaluation turn-limit bug")

        winner = final_info.get("winner")

        if winner is not None:
            winner_colour = env.env.agent_colours[winner]

            if winner_colour == env.model_colour:
                wins += 1
            else:
                opponent_wins += 1

        if final_truncated:
            truncations += 1

        total_model_turns += env.env.turns_per_player[model_agent]

        target_colour = env.env.board.colour_opposites[env.model_colour]
        target_cells = set(env.env.board.axial_of_colour(target_colour))

        start_cells = set(env.env.board.axial_of_colour(env.model_colour))

        own_pins = [
            p for p in env.env.boardPins
            if p.color == env.model_colour
        ]

        pieces_in_target = sum(
            1 for p in own_pins
            if p.axialindex in target_cells
        )

        pieces_in_start = sum(
            1 for p in own_pins
            if p.axialindex in start_cells
        )

        total_pieces_in_target += pieces_in_target
        total_pieces_in_start += pieces_in_start

    print(
        f"\n===== EVALUATION =====\n"
        f"Games: {num_games}\n"
        f"Model win rate: {wins / num_games:.2%}\n"
        f"Opponent win rate: {opponent_wins / num_games:.2%}\n"
        f"Truncation rate: {truncations / num_games:.2%}\n"
        f"Avg reward: {total_reward / num_games:.3f}\n"
        f"Avg model turns: {total_model_turns / num_games:.2f}\n"
        f"Avg own pieces in target: {total_pieces_in_target / num_games:.2f}/10\n"
        f"Avg own pieces still in start: {total_pieces_in_start / num_games:.2f}/10\n"
        f"======================\n"
    )

    metrics = {
        "win_rate": wins / num_games,
        "opponent_win_rate": opponent_wins / num_games,
        "truncation_rate": truncations / num_games,
        "avg_reward": total_reward / num_games,
        "avg_model_turns": total_model_turns / num_games,
        "avg_target_pieces": total_pieces_in_target / num_games,
        "avg_start_pieces": total_pieces_in_start / num_games,
    }

    return metrics



def train_phases(
    total_timesteps,
    max_turns_per_player,
    recent_positions_per_pin,
    cycles,
    save_metrics,
    min_opponents,
    max_opponents,
    self_play=False
):
    training_phases = PHASES * cycles

    phase_steps = total_timesteps // len(training_phases)
    remainder = total_timesteps % len(training_phases)

    ROOT = os.getcwd() 
    checkpoint_dir = os.path.join(ROOT, CHECKPOINT_DIR_NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = os.path.join(ROOT, "ppo_no_progress_reward12_mlp.zip")
    latest_vecnormalize_path = os.path.join(ROOT, "vecnormalize_mlp.pkl")

    best_model_path = os.path.join(checkpoint_dir, "ppo_best.zip")
    best_vecnormalize_path = os.path.join(checkpoint_dir, "vecnormalize_best.pkl")

    model = None

    metrics_callback = (
        MetricsLoggerCallback(csv_path="training_metrics.csv")
        if save_metrics
        else None
    )

    if os.path.exists(model_path):
        t = time.time()
        print(f"Loading existing model: {model_path}")
        model = MaskablePPO.load(model_path)
        print("Loaded main model in", time.time() - t, "seconds")
    else:
        print("No existing model found, starting fresh training.")

    print("Using MLP policy with flattened observations.")
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(
            pi=[512, 256],
            vf=[512, 256],
        )
    )

    def load_checkpoint_opponents():
        loaded = []

        if not self_play:
            return loaded

        checkpoint_paths = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if f.startswith("ppo_cycle_") and f.endswith(".zip")
        ]

        checkpoint_paths = sorted(checkpoint_paths)[-5:]

        for path in checkpoint_paths:
            filename = os.path.basename(path)
            cycle_id = filename.replace("ppo_cycle_", "").replace(".zip", "")

            vec_path = os.path.join(
                checkpoint_dir,
                f"vecnormalize_cycle_{cycle_id}.pkl"
            )

            checkpoint_model = MaskablePPO.load(path, device="cpu")

            checkpoint_vecnorm = None
            if os.path.exists(vec_path):
                dummy_env = DummyVecEnv([
                    make_env(
                        model_colour="red",
                        opponent_colour="blue",
                        max_turns_per_player=max_turns_per_player,
                        recent_positions_per_pin=recent_positions_per_pin,
                        min_opponents=min_opponents,
                        max_opponents=max_opponents,
                        loaded_checkpoints=[],
                    )
                ])

                checkpoint_vecnorm = VecNormalize.load(vec_path, dummy_env)
                checkpoint_vecnorm.training = False
                checkpoint_vecnorm.norm_reward = False

            loaded.append((checkpoint_model, checkpoint_vecnorm))

        return loaded

    n_envs = 8

    t = time.time()
    print("Loading checkpoint opponents...")
    loaded_checkpoints = load_checkpoint_opponents()
    print(
        f"Loaded {len(loaded_checkpoints)} checkpoint opponents in",
        time.time() - t,
        "seconds"
    )

    best_score = -float("inf")
    patience = 6
    bad_cycles = 0
    vec_env = None

    for phase_index, (model_colour, opponent_colour) in enumerate(training_phases):
        steps = phase_steps + (1 if phase_index < remainder else 0)

        if vec_env is not None:
            vec_env.close()

        t = time.time()
        print("Creating vec_env...")

        vec_env = SubprocVecEnv([
            make_env(
                model_colour=model_colour,
                opponent_colour=opponent_colour,
                max_turns_per_player=max_turns_per_player,
                recent_positions_per_pin=recent_positions_per_pin,
                min_opponents=min_opponents,
                max_opponents=max_opponents,
                loaded_checkpoints=loaded_checkpoints,
            )
            for _ in range(n_envs)
        ])

        print("Vec_env created in", time.time() - t, "seconds")

        if os.path.exists(latest_vecnormalize_path):
            print(f"Loading latest VecNormalize stats: {latest_vecnormalize_path}")
            vec_env = VecNormalize.load(latest_vecnormalize_path, vec_env)
            vec_env.training = True
            vec_env.norm_obs = True
            vec_env.norm_reward = True
            vec_env.clip_reward = 10.0
        else:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_reward=10.0,
            )

        current_cycle = phase_index // len(PHASES)
        ent_coef = max(0.005, 0.03 * (0.96 ** current_cycle))
        print(f"ent_coef = {ent_coef:.5f}")

        if model is None:
            model = MaskablePPO(
                POLICY_TYPE,
                vec_env,

                learning_rate=1e-4,
                clip_range=0.20,
                n_epochs=4,

                gamma=0.98,
                gae_lambda=0.95,

                n_steps=512,
                batch_size=512,

                ent_coef=ent_coef,
                vf_coef=0.3,
                max_grad_norm=0.5,
                target_kl=0.03,

                policy_kwargs=policy_kwargs,
                verbose=1,
            )

        else:
            model.set_env(vec_env)

            model.learning_rate = FloatSchedule(1e-4)
            model.lr_schedule = FloatSchedule(1e-4)
            model.clip_range = FloatSchedule(0.20)
            model.n_epochs = 4

            model.gamma = 0.98
            model.gae_lambda = 0.95
            model.batch_size = 512

            model.ent_coef = ent_coef

            model.vf_coef = 0.3
            model.max_grad_norm = 0.5
            model.target_kl = 0.03

        print(
            f"Training phase {phase_index + 1}/{len(training_phases)}: "
            f"{model_colour} model vs {opponent_colour} heuristic "
            f"for {steps} steps"
        )

        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=metrics_callback,
        )

        phases_per_cycle = len(PHASES)

        if (phase_index + 1) % phases_per_cycle == 0:
            cycle_number = (phase_index + 1) // phases_per_cycle

            cycle_model_path = os.path.join(
                checkpoint_dir,
                f"ppo_cycle_{cycle_number}.zip"
            )

            cycle_vecnormalize_path = os.path.join(
                checkpoint_dir,
                f"vecnormalize_cycle_{cycle_number}.pkl"
            )

            model.save(cycle_model_path)
            vec_env.save(cycle_vecnormalize_path)

            model.save(model_path)
            vec_env.save(latest_vecnormalize_path)

            print(f"Saved cycle checkpoint: {cycle_model_path}")
            print(f"Saved cycle VecNormalize stats: {cycle_vecnormalize_path}")
            print(f"Updated latest VecNormalize stats: {latest_vecnormalize_path}")

            print(f"\nEvaluating cycle {cycle_number} with 100 games...")

            vec_env.training = False
            vec_env.norm_reward = False

            metrics = evaluate_model(
                model,
                vec_env=vec_env,
                num_games=100,
            )

            vec_env.training = True
            vec_env.norm_reward = True

            score = (
                2.0 * metrics["win_rate"]
                - 0.5 * metrics["truncation_rate"]
                - 0.005 * metrics["avg_model_turns"]
                + 0.02 * metrics["avg_target_pieces"]
            )

            print(f"Cycle score: {score:.4f}")
            print(f"Best score: {best_score:.4f}")

            if score > best_score:
                best_score = score
                bad_cycles = 0

                model.save(best_model_path)
                vec_env.save(best_vecnormalize_path)

                print(f"New best model saved: {best_model_path}")
                print(f"New best VecNormalize saved: {best_vecnormalize_path}")

            else:
                bad_cycles += 1
                print(f"No improvement for {bad_cycles}/{patience} cycles")

                if bad_cycles >= patience:
                    print("Early stopping: no improvement.")
                    break

            if self_play:
                t = time.time()
                print("Reloading checkpoint opponents with matching VecNormalize stats...")
                loaded_checkpoints = load_checkpoint_opponents()
                print(
                    f"Reloaded {len(loaded_checkpoints)} checkpoint opponents in",
                    time.time() - t,
                    "seconds"
                )

    model.save(model_path)

    if vec_env is not None:
        vec_env.save(latest_vecnormalize_path)
        vec_env.close()

    print(f"Final model saved: {model_path}")
    print(f"Final latest VecNormalize saved: {latest_vecnormalize_path}")

    return model


# ==================== Training configuration ====================

DefaultCycles = 50
TimeSteps_per_phase = 50_000
TimeSteps = TimeSteps_per_phase * len(PHASES) * DefaultCycles

TIMESTEPS = TimeSteps
MAX_TURNS_PER_PLAYER = 100
RECENT_POSITIONS_PER_PIN = 2
CYCLES = DefaultCycles
MIN_OPPONENTS = 1
MAX_OPPONENTS = 5
USE_SELF_PLAY = True
SAVE_METRICS = True


def main():
    print("Training configuration:")
    print("OBS_MODE=mlp")
    print(f"POLICY_TYPE={POLICY_TYPE}")
    print(f"TIMESTEPS={TIMESTEPS}")
    print(f"CYCLES={CYCLES}")
    print(f"MIN_OPPONENTS={MIN_OPPONENTS}, MAX_OPPONENTS={MAX_OPPONENTS}")

    model = train_phases(
        total_timesteps=TIMESTEPS,
        max_turns_per_player=MAX_TURNS_PER_PLAYER,
        recent_positions_per_pin=RECENT_POSITIONS_PER_PIN,
        cycles=CYCLES,
        min_opponents=MIN_OPPONENTS,
        max_opponents=MAX_OPPONENTS,
        save_metrics=SAVE_METRICS,
        self_play=USE_SELF_PLAY,
    )

    return model


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
