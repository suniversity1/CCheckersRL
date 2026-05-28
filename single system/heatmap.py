import os
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import Env

from pettingzoo_checkers import CheckersParallelEnv



ROOT = os.getcwd()

model1_path = os.path.join(ROOT, "single system", "saved_models", "model1.zip")
model1_vecnormalize_path = os.path.join(ROOT, "single system", "saved_models", "model1.pkl")



SNAPSHOT_TURNS = [20, 40, 60]
N_GAMES = 100


def hex_distance(q1, r1, q2, r2):
    s1 = -q1 - r1
    s2 = -q2 - r2
    return max(abs(q1 - q2), abs(r1 - r2), abs(s1 - s2))


def choose_heuristic_action(env, colour, epsilon=0.05, top_k=2):
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


class VecNormalizeLoaderEnv(Env):
    def __init__(self, players, max_turns_per_player, recent_positions_per_pin):
        super().__init__()

        self.env = CheckersParallelEnv(
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


def load_model_and_vec(model_path, vecnormalize_path, players):
    model = MaskablePPO.load(model_path)

    dummy_vec_env = DummyVecEnv([
        lambda: VecNormalizeLoaderEnv(
            players=players,
            max_turns_per_player=100,
            recent_positions_per_pin=2,
        )
    ])

    vecnormalize = VecNormalize.load(vecnormalize_path, dummy_vec_env)
    vecnormalize.training = False
    vecnormalize.norm_reward = False

    return model, vecnormalize


def get_occupancy_vector(env, colour):
    values = np.zeros(env.num_cells, dtype=np.float32)

    for pin in env.boardPins:
        if pin.color == colour:
            values[pin.axialindex] += 1.0

    return values


def run_snapshot_heatmaps(
    model,
    vecnormalize,
    n_games=100,
    players=None,
    model_colour="red",
    snapshot_turns=None,
    heuristic_epsilon=0.05,
    heuristic_top_k=2,
    max_turns_per_player=100,
    recent_positions_per_pin=2,
):
    if players is None:
        players = ["red", "blue"]

    if snapshot_turns is None:
        snapshot_turns = [20, 40, 60]

    snapshot_turns = sorted(snapshot_turns)

    snapshots = {turn: None for turn in snapshot_turns}
    counts = {turn: 0 for turn in snapshot_turns}

    for game_idx in range(n_games):
        print(f"  Game {game_idx + 1}/{n_games}")

        env = CheckersParallelEnv(
            num_players=len(players),
            max_turns_per_player=max_turns_per_player,
            recent_positions_per_pin=recent_positions_per_pin,
            player_colours=players,
        )

        env.reset()

        model_turn_count = 0
        recorded = set()

        while True:
            current_agent = env.current_agent
            current_colour = env.agent_colours[current_agent]

            if current_colour == model_colour:
                obs = env._observe(current_agent)
                obs = vecnormalize.normalize_obs(np.array([obs]))[0]

                mask = env.action_mask(current_colour, normalized=True)

                norm_action, _ = model.predict(
                    obs,
                    action_masks=mask,
                    deterministic=True,
                )

                env.step_normalized({current_agent: int(norm_action)})
                model_turn_count += 1

                if (
                    model_turn_count in snapshot_turns
                    and model_turn_count not in recorded
                ):
                    occ = get_occupancy_vector(env, model_colour)

                    if snapshots[model_turn_count] is None:
                        snapshots[model_turn_count] = np.zeros_like(occ)

                    snapshots[model_turn_count] += occ
                    counts[model_turn_count] += 1
                    recorded.add(model_turn_count)

            else:
                real_action = choose_heuristic_action(
                    env,
                    current_colour,
                    epsilon=heuristic_epsilon,
                    top_k=heuristic_top_k,
                )

                if real_action is None:
                    env._advance_turn()
                else:
                    env.step({current_agent: real_action})

            done = env.winner() is not None or any(
                turns >= max_turns_per_player
                for turns in env.turns_per_player.values()
            )

            if done:
                break

            if all(turn in recorded for turn in snapshot_turns):
                break

    for turn in snapshot_turns:
        if snapshots[turn] is None:
            snapshots[turn] = np.zeros(121, dtype=np.float32)
            continue

        if counts[turn] > 0:
            snapshots[turn] /= counts[turn]

    global_max = max(
        snapshot.max()
        for snapshot in snapshots.values()
    )

    if global_max > 0:
        for turn in snapshot_turns:
            snapshots[turn] = snapshots[turn] / global_max

    return snapshots


def plot_board_heatmap(env, values, title, outpath):
    board = env.board

    xs = np.array([cell.x for cell in board.cells])
    ys = np.array([cell.y for cell in board.cells])

    values = np.asarray(values, dtype=np.float32)

    plt.figure(figsize=(8, 8))

    plt.scatter(
        xs,
        ys,
        s=520,
        c=values,
        cmap="Oranges",
        edgecolors="black",
        linewidths=0.8,
        vmin=0,
        vmax=1,
    )

    for cell in board.cells:
        if cell.postype != "board":
            plt.text(
                cell.x,
                cell.y,
                cell.postype[0].upper(),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )

    plt.colorbar(label="Normalized peg-location frequency")
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    players = ["red", "blue"]

    pairs = [
        (model1_path, model1_vecnormalize_path, "model1"),
    ]

    os.makedirs("heatmaps", exist_ok=True)

    env_for_plot = CheckersParallelEnv(
        num_players=2,
        max_turns_per_player=100,
        recent_positions_per_pin=2,
        player_colours=players,
    )
    env_for_plot.reset()

    for model_path, vec_path, name in pairs:
        print(f"Running {name} model vs heuristic...")

        model, vecnormalize = load_model_and_vec(model_path, vec_path, players)

        snapshots = run_snapshot_heatmaps(
            model=model,
            vecnormalize=vecnormalize,
            n_games=N_GAMES,
            players=players,
            model_colour="red",
            snapshot_turns=SNAPSHOT_TURNS,
            heuristic_epsilon=0.05,
            heuristic_top_k=2,
            max_turns_per_player=100,
            recent_positions_per_pin=2,
        )

        for turn, heatmap_values in snapshots.items():
            out = os.path.join(
                "heatmaps",
                f"heatmap_{name}_turn_{turn}_vs_heuristic_board.png",
            )

            plot_board_heatmap(
                env_for_plot,
                heatmap_values,
                f"{name}: peg locations after {turn} model turns",
                out,
            )

            print(f"Saved: {out}")


if __name__ == "__main__":
    main()