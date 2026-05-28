import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from sb3_contrib import MaskablePPO

from checkers_gui import BoardGUI
from pettingzoo_checkers import CheckersParallelEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import Env


CheckersEnvironment = CheckersParallelEnv
CheckerEnvType = CheckersParallelEnv


def choose_random_action(env: CheckerEnvType, colour: str) -> Optional[int]:
    legal = env.legal_actions(colour, normalized=False)
    if not legal:
        return None
    return int(np.random.choice(legal))


def choose_heuristic_action(
    env: CheckerEnvType,
    colour: str,
    epsilon: float = 0.05,
    top_k: int = 5,
) -> Optional[int]:
    legal = env.legal_actions(colour, normalized=False)

    if not legal:
        return None

    if np.random.rand() < epsilon:
        return int(np.random.choice(legal))

    scored = []

    target_colour = env.board.colour_opposites[colour]
    target_cells = env.board.axial_of_colour(target_colour)
    target_set = set(target_cells)

    for action in legal:
        pin_id, dest = env.decode_action(action)
        pin = env._pin(colour, pin_id)

        if pin is None:
            continue

        old_cell = env.board.cells[pin.axialindex]
        new_cell = env.board.cells[dest]

        old_dist = min(
            hex_distance(
                old_cell.q,
                old_cell.r,
                env.board.cells[t].q,
                env.board.cells[t].r,
            )
            for t in target_cells
        )

        new_dist = min(
            hex_distance(
                new_cell.q,
                new_cell.r,
                env.board.cells[t].q,
                env.board.cells[t].r,
            )
            for t in target_cells
        )

        score = old_dist - new_dist

        move_dist = hex_distance(
            old_cell.q,
            old_cell.r,
            new_cell.q,
            new_cell.r,
        )

        if move_dist >= 2:
            score += 0.25 * move_dist

        if dest in target_set:
            score += 0.5

        scored.append((score, action))

    if not scored:
        return int(np.random.choice(legal))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[: min(top_k, len(scored))]

    return int(np.random.choice([action for _, action in top]))


def hex_distance(q1, r1, q2, r2):
    s1 = -q1 - r1
    s2 = -q2 - r2
    return max(abs(q1 - q2), abs(r1 - r2), abs(s1 - s2))


class VecNormalizeLoaderEnv(Env):
    def __init__(
        self,
        players,
        max_turns_per_player,
        recent_positions_per_pin,
    ):
        super().__init__()

        self.env = CheckersEnvironment(
            num_players=len(players),
            max_turns_per_player=max_turns_per_player,
            recent_positions_per_pin=recent_positions_per_pin,
            player_colours=players,
        )

        self.env.reset()

        first_agent = self.env.agents[0]

        self.observation_space = self.env.observation_spaces[first_agent]
        self.action_space = self.env.action_spaces[first_agent]

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        first_agent = self.env.agents[0]
        return self.env._observe(first_agent), {}

    def step(self, action):
        first_agent = self.env.agents[0]
        obs = self.env._observe(first_agent)
        return obs, 0.0, False, False, {}


def play_game_gui(
    model_path: str,
    vecnormalize_path: Optional[str] = None,
    players: Optional[List[str]] = None,
    player_modes: Optional[Dict[str, str]] = None,
    max_turns_per_player: int = 300,
    recent_positions_per_pin: int = 3,
    move_delay: float = 0.2,
    heuristic_epsilon: float = 0.05,
    heuristic_top_k: int = 5,
):
    if players is None:
        players = ["red", "blue"]

    if player_modes is None:
        player_modes = {players[0]: "model"}
        for colour in players[1:]:
            player_modes[colour] = "heuristic"

    model_path = Path(model_path)

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model from {model_path}...")
    print("Using observation mode: MLP")
    model = MaskablePPO.load(str(model_path))
    print("Model loaded.")

    env = CheckersEnvironment(
        num_players=len(players),
        max_turns_per_player=max_turns_per_player,
        recent_positions_per_pin=recent_positions_per_pin,
        player_colours=players,
    )

    env.reset()

    vecnormalize = None

    if vecnormalize_path is not None:
        vecnormalize_path = Path(vecnormalize_path)

        if vecnormalize_path.exists():
            print(f"Loading VecNormalize from {vecnormalize_path}...")

            dummy_vec_env = DummyVecEnv([
                lambda: VecNormalizeLoaderEnv(
                    players=players,
                    max_turns_per_player=max_turns_per_player,
                    recent_positions_per_pin=recent_positions_per_pin,
                )
            ])

            vecnormalize = VecNormalize.load(str(vecnormalize_path), dummy_vec_env)
            vecnormalize.training = False
            vecnormalize.norm_reward = False

            print("VecNormalize loaded.")
        else:
            print(f"VecNormalize file not found: {vecnormalize_path}")

    gui = BoardGUI(env.board, env.boardPins)

    move_count = 0
    total_rewards = {agent: 0.0 for agent in env.agents}

    print("\n" + "=" * 70)
    print(f"Game started with {len(players)} players")
    for agent in env.agents:
        colour = env.agent_colours[agent]
        mode = player_modes.get(colour, "heuristic")
        print(f"{agent:10s} | {colour:12s} -> {mode}")
    print("=" * 70 + "\n")

    while True:
        current_agent = env.current_agent
        current_colour = env.agent_colours[current_agent]
        mode = player_modes.get(current_colour, "heuristic").lower()

        if mode == "model":
            obs = env._observe(current_agent)

            if vecnormalize is not None:
                obs = vecnormalize.normalize_obs(
                    np.array([obs])
                )[0]

            mask = env.action_mask(
                current_colour,
                normalized=True,
            )

            if mask.sum() == 0:
                print(f"{current_colour.upper()} has no legal normalized actions.")
                env._advance_turn()
                continue

            norm_action, _ = model.predict(
                obs,
                action_masks=mask,
                deterministic=True,
            )

            norm_action = int(norm_action)

            pin_id, real_dest = env.decode_normalized_action(
                current_colour,
                norm_action,
            )

            real_action = env.encode_action(pin_id, real_dest)

            _, rewards, terminations, truncations, infos = env.step_normalized(
                {current_agent: norm_action}
            )

            move_count += 1
            total_rewards[current_agent] += rewards.get(current_agent, 0.0)

            info = infos.get(current_agent, {})

            print(
                f"Move {move_count:4d} "
                f"[{current_colour.upper():12s} MODEL] "
                f"Pin {pin_id} -> Cell {real_dest} "
                f"| norm_action={norm_action} "
                f"| reward={rewards.get(current_agent, 0.0):.3f}"
            )

        elif mode == "random":
            real_action = choose_random_action(env, current_colour)

            if real_action is None:
                print(f"{current_colour.upper()} has no legal actions.")
                env._advance_turn()
                continue

            pin_id, real_dest = env.decode_action(real_action)

            _, rewards, terminations, truncations, infos = env.step(
                {current_agent: real_action}
            )

            move_count += 1
            total_rewards[current_agent] += rewards.get(current_agent, 0.0)

            print(
                f"Move {move_count:4d} "
                f"[{current_colour.upper():12s} RANDOM] "
                f"Pin {pin_id} -> Cell {real_dest}"
            )

        else:
            real_action = choose_heuristic_action(
                env,
                current_colour,
                epsilon=heuristic_epsilon,
                top_k=heuristic_top_k,
            )

            if real_action is None:
                print(f"{current_colour.upper()} has no legal actions.")
                env._advance_turn()
                continue

            pin_id, real_dest = env.decode_action(real_action)

            _, rewards, terminations, truncations, infos = env.step(
                {current_agent: real_action}
            )

            move_count += 1
            total_rewards[current_agent] += rewards.get(current_agent, 0.0)

            print(
                f"Move {move_count:4d} "
                f"[{current_colour.upper():12s} HEURISTIC] "
                f"Pin {pin_id} -> Cell {real_dest}"
            )

        gui.refresh(env.boardPins)
        gui.root.update()
        time.sleep(move_delay)

        done = any(terminations.values()) or any(truncations.values())
        winner = env.winner()

        if done:
            print("\n" + "=" * 70)

            if winner is not None:
                winner_colour = env.agent_colours[winner]
                print(f"WINNER: {winner_colour.upper()} ({winner})")
            else:
                print("Game ended by turn limit.")

            print("\nFinal rewards:")
            for agent in env.agents:
                colour = env.agent_colours[agent]
                print(
                    f"{agent:10s} | {colour:12s}: "
                    f"{total_rewards[agent]:8.3f}"
                )

            print(f"\nTotal moves: {move_count}")
            print("=" * 70)
            break

    print("\nWindow will close when you exit the GUI.")
    gui.run()


ROOT = os.getcwd()
model_path = os.path.join(ROOT, "single system", "saved_models",  "model1.zip")
vecnormalize_path = os.path.join(ROOT, "single system", "saved_models",  "model1.pkl")

if __name__ == "__main__":
    play_game_gui(
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        players=["red", "blue", "lawn green", "gray0"],
        player_modes={
            "red": "model",
            "blue": "model",
            "lawn green": "model",
            "gray0": "model",
        },
        max_turns_per_player=100,
        recent_positions_per_pin=2,
        move_delay=0.1,
        heuristic_epsilon=0.05,
        heuristic_top_k=2,
    )