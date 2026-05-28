import os
import json
import socket
import time
import random
from typing import Dict, Any, Optional, Tuple

import numpy as np
import math
from collections import deque
from gymnasium import Env
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pettingzoo_checkers import CheckersParallelEnv


HOST = "127.0.0.1"
PORT = 50555

ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(ROOT, "saved_models", "ppo_no_progress_reward10_mlp.zip")
VECNORMALIZE_PATH = os.path.join(ROOT, "saved_models", "vecnormalize_ppo_no_progress_reward10_mlp.pkl")
DEBUG_LOCAL_ADAPTER = False


def rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10.0)

    try:
        s.connect((HOST, PORT))
    except Exception as e:
        return {"ok": False, "error": f"connect-failed: {e}"}

    s.sendall(json.dumps(payload).encode("utf-8"))

    chunks = []
    while True:
        chunk = s.recv(65_536)
        if not chunk:
            break
        chunks.append(chunk)

    s.close()

    if not chunks:
        return {"ok": False, "error": "no-response"}

    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad-json: {e}"}


class VecNormalizeLoaderEnv(Env):
    def __init__(self, players):
        super().__init__()

        self.env = CheckersParallelEnv(
            num_players=len(players),
            max_turns_per_player=600,
            recent_positions_per_pin=2,
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


class MLPModelPlayer:
    def __init__(self, model_path: str, vecnormalize_path: str):
        print(f"Loading model: {model_path}")
        self.model = MaskablePPO.load(model_path)
        self.vecnormalize_path = vecnormalize_path
        self.vecnormalize = None
        self.vec_players = None
        self.recent_pin_positions = {}
        print("Model loaded.")

    # --- Local adapter helpers copied from test_two_ai_players_local.py ---
    def _ensure_history_for_state(self, state: Dict[str, Any]):
        for player in state["players"]:
            colour = player["colour"]
            for pin_id, pos in enumerate(state["pins"].get(colour, [])):
                key = (colour, int(pin_id))
                history = self.recent_pin_positions.get(key)

                if history is None:
                    self.recent_pin_positions[key] = deque(
                        [int(pos)],
                        maxlen=2,
                    )
                    continue

                current_pos = int(pos)
                if not history or history[-1] != current_pos:
                    history.append(current_pos)

    def _apply_last_move_to_history(self, state: Dict[str, Any]):
        last_move = state.get("last_move")
        if not last_move:
            return

        colour = last_move.get("colour")
        pin_id = last_move.get("pin_id")
        dest = last_move.get("to")

        if colour is None or pin_id is None or dest is None:
            return

        key = (colour, int(pin_id))
        history = self.recent_pin_positions.get(key)

        if history is None:
            history = deque(maxlen=2)
            self.recent_pin_positions[key] = history

        dest = int(dest)
        if not history or history[-1] != dest:
            history.append(dest)

    def _target_centroid(self, env: CheckersParallelEnv, colour: str):
        target_colour = env.board.colour_opposites[colour]
        target_real_indices = env.board.axial_of_colour(target_colour)

        target_norm_cells = [env.board.cells[env._norm_index(colour, idx)] for idx in target_real_indices]

        centroid_x = sum(cell.x for cell in target_norm_cells) / len(target_norm_cells)
        centroid_y = sum(cell.y for cell in target_norm_cells) / len(target_norm_cells)

        return centroid_x, centroid_y

    def _is_forward_move(self, env: CheckersParallelEnv, colour: str, pin, dest: int):
        steps = env._rotation_steps_for_colour(colour)

        cur = env.board.cells[pin.axialindex]
        dst = env.board.cells[int(dest)]

        cq, cr = env._rotate_axial_60(cur.q, cur.r, steps)
        dq, dr = env._rotate_axial_60(dst.q, dst.r, steps)

        target_colour = env.board.colour_opposites[colour]
        target_indices = env.board.axial_of_colour(target_colour)

        def hexd(q1, r1, q2, r2):
            s1 = -q1 - r1
            s2 = -q2 - r2
            return max(abs(q1 - q2), abs(r1 - r2), abs(s1 - s2))

        min_cur = min(
            hexd(cq, cr, *env._rotate_axial_60(env.board.cells[idx].q, env.board.cells[idx].r, steps))
            for idx in target_indices
        )

        min_dst = min(
            hexd(dq, dr, *env._rotate_axial_60(env.board.cells[idx].q, env.board.cells[idx].r, steps))
            for idx in target_indices
        )

        # If inside goal (min_cur == 0) compare pixel distance to centroid,
        # otherwise use hex-distance. Always require strictly closer.
        if min_cur == 0:
            cur_norm_idx = env._norm_index(colour, pin.axialindex)
            dst_norm_idx = env._norm_index(colour, int(dest))

            cur_cell = env.board.cells[cur_norm_idx]
            dst_cell = env.board.cells[dst_norm_idx]

            cx, cy = self._target_centroid(env, colour)

            cur_dist = math.hypot(cur_cell.x - cx, cur_cell.y - cy)
            dst_dist = math.hypot(dst_cell.x - cx, dst_cell.y - cy)

            forward = dst_dist < cur_dist
        else:
            forward = min_dst < min_cur

        if DEBUG_LOCAL_ADAPTER:
            print(f"[DEBUG] player._is_forward_move: colour={colour} pin={getattr(pin,'id',None)} cur={pin.axialindex} dst={dest} min_cur={min_cur} min_dst={min_dst} forward={forward}")
        return forward

    def _is_allowed_destination(self, env: CheckersParallelEnv, colour: str, dest: int, pin_id: Optional[int] = None):
        cell_type = env.board.cells[int(dest)].postype
        target_colour = env.board.colour_opposites[colour]

        if pin_id is not None:
            pin = next((p for p in env.boardPins if p.color == colour and p.id == int(pin_id)), None)

            if pin is None:
                if DEBUG_LOCAL_ADAPTER:
                    print(f"[DEBUG] player._is_allowed_destination: pin not found colour={colour} pin_id={pin_id} dest={dest}")
                return False

            current_cell = env.board.cells[pin.axialindex]
            dest_cell = env.board.cells[int(dest)]

            # If already in target zone, cannot leave.
            if current_cell.postype == target_colour:
                if dest_cell.postype != target_colour:
                    if DEBUG_LOCAL_ADAPTER:
                        print(f"[DEBUG] player._is_allowed_destination: leave_target_disallowed colour={colour} pin={pin.id} cur={pin.axialindex} dest={dest} cur_type={current_cell.postype} dest_type={dest_cell.postype}")
                    return False

                # inside target: must strictly reduce centroid distance
                forward = self._is_forward_move(env, colour, pin, dest)
                if DEBUG_LOCAL_ADAPTER:
                    print(f"[DEBUG] player._is_allowed_destination: inside_target check colour={colour} pin={pin.id} cur={pin.axialindex} dest={dest} forward={forward}")
                return forward

            # Outside target: enforce strict forward-only
            forward = self._is_forward_move(env, colour, pin, dest)
            if not forward:
                if DEBUG_LOCAL_ADAPTER:
                    print(f"[DEBUG] player._is_allowed_destination: not forward colour={colour} pin={pin.id} cur={pin.axialindex} dest={dest}")
                return False

        ok = cell_type in {"board", colour, target_colour}
        if DEBUG_LOCAL_ADAPTER and not ok:
            print(f"[DEBUG] player._is_allowed_destination: cell_type_blocked colour={colour} dest={dest} type={cell_type}")
        return ok

    def _recent_positions_for_pin(self, colour: str, pin_id: int):
        history = self.recent_pin_positions.get((colour, int(pin_id)))
        if history is None:
            return set()
        return {int(pos) for pos in history}

    def _pick_non_recent_move(self, env: CheckersParallelEnv, colour: str, legal_moves: Dict[str, Any]):
        candidates = []

        for pid, moves in legal_moves.items():
            pid = int(pid)
            recent_positions = self._recent_positions_for_pin(colour, pid)
            filtered_moves = [
                int(move)
                for move in moves
                if int(move) not in recent_positions and self._is_allowed_destination(env, colour, move, pid)
            ]

            if filtered_moves:
                candidates.append((pid, filtered_moves))

        if not candidates:
            candidates = [
                (int(pid), [int(move) for move in moves])
                for pid, moves in legal_moves.items()
                if moves
            ]

        if not candidates:
            return None

        pid, moves = random.choice(candidates)
        return pid, int(random.choice(moves))
    # --- end helpers ---

    def _load_vecnormalize_if_needed(self, players):
        players = list(players)

        if self.vecnormalize is not None and self.vec_players == players:
            return

        dummy_env = DummyVecEnv([
            lambda: VecNormalizeLoaderEnv(players)
        ])

        self.vecnormalize = VecNormalize.load(self.vecnormalize_path, dummy_env)
        self.vecnormalize.training = False
        self.vecnormalize.norm_reward = False
        self.vec_players = players

    def _build_env_from_state(self, state: Dict[str, Any]) -> CheckersParallelEnv:
        players = [
            pl["colour"]
            for pl in state["players"]
        ]

        env = CheckersParallelEnv(
            num_players=len(players),
            max_turns_per_player=600,
            recent_positions_per_pin=2,
            player_colours=players,
        )

        env.reset()

        # Clear board occupancy
        for cell in env.board.cells:
            cell.occupied = False

        # Apply server pin positions
        pins = state["pins"]

        for agent in env.agents:
            colour = env.agent_colours[agent]
            server_positions = pins[colour]

            for pin, pos in zip(env.agent_pins[agent], server_positions):
                pin.axialindex = int(pos)
                env.board.cells[int(pos)].occupied = True

        # Match current turn
        current_colour = state.get("current_turn_colour")
        for agent in env.agents:
            if env.agent_colours[agent] == current_colour:
                env.current_agent = agent
                break

        env._clear_action_cache()
        return env

    def choose_move(
        self,
        state: Dict[str, Any],
        my_colour: str,
        legal_moves: Dict[str, Any],
    ) -> Optional[Tuple[int, int]]:
        # Keep a local history and apply last move to it so fallback and
        # masking can avoid back-and-forth.
        state = dict(state)
        self._ensure_history_for_state(state)
        self._apply_last_move_to_history(state)

        env = self._build_env_from_state(state)

        # Populate env.pin_history from our recent_pin_positions so the
        # environment's mask logic can consider recent positions.
        for player in state["players"]:
            colour = player["colour"]
            for pin_id, pos in enumerate(state["pins"].get(colour, [])):
                key = (colour, int(pin_id))
                history = self.recent_pin_positions.get(key)
                if history is None:
                    history = deque([int(pos)], maxlen=2)
                    self.recent_pin_positions[key] = history

                env.pin_history[key] = deque(history, maxlen=2)

        # Inject server legal moves into PettingZoo env so normalized mask
        # corresponds to server legal moves.
        current_colour = state.get("current_turn_colour")
        server_actions_real = []
        for pid, moves in legal_moves.items():
            pid = int(pid)
            for real_dest in moves:
                real_dest = int(real_dest)
                server_actions_real.append(env.encode_action(pid, real_dest))

        env._legal_cache[(current_colour, False)] = server_actions_real

        server_actions_norm = []
        for real_action in server_actions_real:
            pin_id = int(real_action) // env.num_cells
            real_dest = int(real_action) % env.num_cells
            norm_action = env.encode_normalized_action(current_colour, pin_id, real_dest)
            server_actions_norm.append(norm_action)

        env._legal_cache[(current_colour, True)] = server_actions_norm

        players = [pl["colour"] for pl in state["players"]]
        self._load_vecnormalize_if_needed(players)

        model_agent = None
        for agent in env.agents:
            if env.agent_colours[agent] == my_colour:
                model_agent = agent
                break

        if model_agent is None:
            return self._fallback_random(legal_moves)

        obs = env._observe(model_agent)
        obs = self.vecnormalize.normalize_obs(np.array([obs]))[0]

        mask = env.action_mask(my_colour, normalized=True)

        if mask.sum() == 0:
            return self._fallback_random(legal_moves)

        norm_action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)

        pin_id, real_dest = env.decode_normalized_action(my_colour, int(norm_action))

        if DEBUG_LOCAL_ADAPTER:
            print(f"[DEBUG] player model predicted norm_action={int(norm_action)} -> pin={pin_id} dest(real)={real_dest}")

        legal_for_pin = legal_moves.get(str(pin_id), legal_moves.get(pin_id, []))
        if DEBUG_LOCAL_ADAPTER:
            print(f"[DEBUG] player legal_for_pin={legal_for_pin}")

        recent_positions = self._recent_positions_for_pin(my_colour, pin_id)

        allowed = (
            int(real_dest) in legal_for_pin
            and int(real_dest) not in recent_positions
            and self._is_allowed_destination(env, my_colour, real_dest, pin_id)
        )

        if DEBUG_LOCAL_ADAPTER:
            print(f"[DEBUG] player recent_positions={recent_positions} allowed={allowed}")

        if allowed:
            self.recent_pin_positions.setdefault((my_colour, int(pin_id)), deque(maxlen=2)).append(int(real_dest))
            return int(pin_id), int(real_dest)

        # Fallback: pick a non-recent allowed move
        fallback_move = self._pick_non_recent_move(env, my_colour, legal_moves)
        if DEBUG_LOCAL_ADAPTER:
            print(f"[DEBUG] player model choice rejected, fallback_move={fallback_move}")

        if fallback_move is None:
            return None

        pid, dest = fallback_move

        self.recent_pin_positions.setdefault((my_colour, int(pid)), deque(maxlen=2)).append(dest)
        return int(pid), dest

    def _fallback_random(self, legal_moves):
        movable = [
            (int(pid), moves)
            for pid, moves in legal_moves.items()
            if moves
        ]

        if not movable:
            return None

        pid, moves = random.choice(movable)
        return int(pid), int(random.choice(moves))


def main():
    print("==== MLP PPO Player ====")

    name = input("Enter name: ").strip()
    if not name:
        return

    agent = MLPModelPlayer(
        model_path=MODEL_PATH,
        vecnormalize_path=VECNORMALIZE_PATH,
    )

    r = rpc({"op": "join", "player_name": name})
    if not r.get("ok"):
        print("JOIN ERROR:", r.get("error"))
        return

    game_id = r["game_id"]
    player_id = r["player_id"]
    colour = r["colour"]

    print(f"Joined game {game_id} as {colour}")

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        status = st.get("state", {}).get("status")
        if status in ("READY_TO_START", "PLAYING"):
            break
        print("Waiting for players...")
        time.sleep(0.5)

    rpc({"op": "start", "game_id": game_id, "player_id": player_id})
    print("Sent START")

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") == "PLAYING":
            break
        time.sleep(0.5)

    print("=== GAME STARTED ===")

    last_move_seen = 0

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            print("State error:", st.get("error"))
            return

        state = st["state"]

        if state["status"] == "FINISHED":
            print("=== GAME FINISHED ===")
            for pl in state["players"]:
                sc = pl.get("score")
                if sc:
                    print(f"{pl['name']} ({pl['colour']}): {sc['final_score']:.1f}")
            break

        if state["move_count"] > last_move_seen:
            mv = state.get("last_move")
            if mv:
                print(
                    f"MOVE: {mv['by']} ({mv['colour']}) "
                    f"{mv['from']} -> {mv['to']}"
                )
            last_move_seen = state["move_count"]

        if state.get("current_turn_colour") == colour:
            legal_req = rpc({
                "op": "get_legal_moves",
                "game_id": game_id,
                "player_id": player_id,
            })

            if not legal_req.get("ok"):
                print("Legal move error:", legal_req.get("error"))
                time.sleep(0.2)
                continue

            move = agent.choose_move(
                state=state,
                my_colour=colour,
                legal_moves=legal_req["legal_moves"],
            )

            if move is None:
                print("No legal move.")
                time.sleep(0.2)
                continue

            pin_id, to_index = move

            print(f"AI move: pin {pin_id} -> {to_index}")

            mv = rpc({
                "op": "move",
                "game_id": game_id,
                "player_id": player_id,
                "pin_id": pin_id,
                "to_index": to_index,
            })

            if not mv.get("ok"):
                print("Move rejected:", mv.get("error"))

        time.sleep(0.2)


if __name__ == "__main__":
    main()