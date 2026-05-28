import importlib.util
import os
from collections import deque

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv
from scipy.optimize import linear_sum_assignment


def _load_module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hex_distance(q1, r1, q2, r2):
    s1 = -q1 - r1
    s2 = -q2 - r2
    return max(abs(q1 - q2), abs(r1 - r2), abs(s1 - s2))


class CheckersParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_players=2, max_turns_per_player=200, recent_positions_per_pin=3, player_colours=None):
        super().__init__()
        root = os.path.dirname(__file__)
        board_mod = _load_module_from_path(os.path.join(root, "checkers_board.py"), "checkers_board")
        pins_mod = _load_module_from_path(os.path.join(root, "checkers_pins.py"), "checkers_pins")

        self.HexBoard = board_mod.HexBoard
        self.Pin = pins_mod.Pin

        self.num_players = int(num_players)
        self.max_turns_per_player = int(max_turns_per_player)
        self.recent_positions_per_pin = max(0, int(recent_positions_per_pin))
        self.player_colours = list(player_colours) if player_colours is not None else None

        self._build_game()

    def _build_game(self):
        self.board = self.HexBoard(R=4, hole_radius=16, spacing=34)

        half_colours1 = ["red", "lawn green", "yellow"]
        half_colours2 = ["blue", "gray0", "purple"]

        self._legal_cache = {}
        self._mask_cache = {}

        if self.player_colours is not None:
            chosen = list(self.player_colours)
        else:
            chosen = []
            for i in range(self.num_players):
                pair_idx = (i // 2) % len(half_colours1)
                if i % 2 == 0:
                    chosen.append(half_colours1[pair_idx])
                else:
                    chosen.append(half_colours2[pair_idx])

        self.agents = [f"player_{i}" for i in range(self.num_players)]
        self.possible_agents = self.agents.copy()
        self.agent_colours = {agent: chosen[i] for i, agent in enumerate(self.agents)}

        self.num_cells = len(self.board.cells)
        self._build_perspective_maps()
        self.num_pins_per_player = 10
        self._action_space = spaces.Discrete(self.num_pins_per_player * self.num_cells)
        self.action_spaces = {agent: self._action_space for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(self.num_cells * 8,), dtype=np.float32)
            for agent in self.agents
        }

        self.agent_pins = {agent: [] for agent in self.agents}
        self.boardPins = []
        for agent in self.agents:
            colour = self.agent_colours[agent]
            axial_cells = self.board.axial_of_colour(colour)
            pins = [self.Pin(self.board, axial_cells[i], id=i, color=colour) for i in range(self.num_pins_per_player)]
            self.agent_pins[agent] = pins
            self.boardPins.extend(pins)

        self.turns_per_player = {agent: 0 for agent in self.agents}
        self.current_agent = self.agents[0]
        self.num_turns = 0
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.pin_history = {}
        for agent in self.agents:
            colour = self.agent_colours[agent]
            for pin in self.agent_pins[agent]:
                history = [pin.axialindex]
                self.pin_history[(colour, pin.id)] = deque(
                    history,
                    maxlen=self.recent_positions_per_pin if self.recent_positions_per_pin > 0 else None,
                )

    def _clear_action_cache(self):
        self._legal_cache.clear()
        self._mask_cache.clear()

    def action_space(self, agent=None):
        return self._action_space

    def observation_space(self, agent=None):
        return spaces.Box(low=0.0, high=1.0, shape=(self.num_cells * 8,), dtype=np.float32)

    def reset(self, seed=None, options=None, return_info=False):
        if seed is not None:
            np.random.seed(seed)
        self._build_game()
        observations = {agent: self._observe(agent) for agent in self.agents}
        if return_info:
            return observations, {agent: {} for agent in self.agents}
        return observations
    
    def _rotation_steps_for_colour(self, colour):
        return {
            "red": 0,
            "blue": 3,
            "lawn green": 2,
            "gray0": 5,
            "yellow": 4,
            "purple": 1,
        }[colour]


    def _rotate_axial_60(self, q, r, steps):
        x = q
        z = r
        y = -x - z

        steps = steps % 6
        for _ in range(steps):
            x, y, z = -z, -x, -y

        return x, z


    def _build_perspective_maps(self):
        self.real_to_norm = {}
        self.norm_to_real = {}

        axial_to_idx = {
            (cell.q, cell.r): idx
            for idx, cell in enumerate(self.board.cells)
        }

        for colour in ["red", "blue", "lawn green", "gray0", "yellow", "purple"]:
            steps = self._rotation_steps_for_colour(colour)

            r2n = {}
            n2r = {}

            for real_idx, cell in enumerate(self.board.cells):
                nq, nr = self._rotate_axial_60(cell.q, cell.r, steps)
                norm_idx = axial_to_idx[(nq, nr)]

                r2n[real_idx] = norm_idx
                n2r[norm_idx] = real_idx

            self.real_to_norm[colour] = r2n
            self.norm_to_real[colour] = n2r


    def _norm_index(self, colour, real_idx):
        return self.real_to_norm[colour][real_idx]


    def _real_index(self, colour, norm_idx):
        return self.norm_to_real[colour][norm_idx]

    def _observe(self, agent):
        """
        Channels:
        0: own pieces
        1: opponent pieces
        2: target cells
        3: source cells
        4: empty valid cells
        5: normalized distance-to-goal
        6: own pins already in target
        7: opponent pins inside my target
        """

        perspective_colour = self.agent_colours[agent]

        target_colour = self.board.colour_opposites[perspective_colour]

        target_cells = set(
            self.board.axial_of_colour(target_colour)
        )

        source_cells = set(
            self.board.axial_of_colour(perspective_colour)
        )

        obs = np.zeros((self.num_cells, 8), dtype=np.float32)

        occupied = set()

        # -------------------------------------------------
        # Occupancy channels
        # -------------------------------------------------

        for pin in self.boardPins:
            occupied.add(pin.axialindex)

            norm_idx = self._norm_index(
                perspective_colour,
                pin.axialindex
            )

            if pin.color == perspective_colour:
                obs[norm_idx, 0] = 1.0
            else:
                obs[norm_idx, 1] = 1.0

        # -------------------------------------------------
        # Static board channels
        # -------------------------------------------------

        for real_idx in range(self.num_cells):
            norm_idx = self._norm_index(
                perspective_colour,
                real_idx
            )

            if real_idx in target_cells:
                obs[norm_idx, 2] = 1.0

            if real_idx in source_cells:
                obs[norm_idx, 3] = 1.0

            if real_idx not in occupied:
                obs[norm_idx, 4] = 1.0

        # -------------------------------------------------
        # Channel 5:
        # normalized distance-to-goal
        # -------------------------------------------------

        target_positions = [
            self.board.cells[idx]
            for idx in target_cells
        ]

        max_possible_distance = 16.0

        for pin in self.boardPins:

            if pin.color != perspective_colour:
                continue

            pin_cell = self.board.cells[pin.axialindex]

            min_dist = min(
                hex_distance(
                    pin_cell.q,
                    pin_cell.r,
                    tgt.q,
                    tgt.r
                )
                for tgt in target_positions
            )

            norm_idx = self._norm_index(
                perspective_colour,
                pin.axialindex
            )

            normalized_dist = 1.0 - (
                min_dist / max_possible_distance
            )

            obs[norm_idx, 5] = normalized_dist

        # -------------------------------------------------
        # Channel 6:
        # own pins already in target
        # -------------------------------------------------

        for pin in self.boardPins:

            if pin.color != perspective_colour:
                continue

            if pin.axialindex in target_cells:

                norm_idx = self._norm_index(
                    perspective_colour,
                    pin.axialindex
                )

                obs[norm_idx, 6] = 1.0

        # -------------------------------------------------
        # Channel 7:
        # opponents inside my target
        # -------------------------------------------------

        for pin in self.boardPins:

            if pin.color == perspective_colour:
                continue

            if pin.axialindex in target_cells:

                norm_idx = self._norm_index(
                    perspective_colour,
                    pin.axialindex
                )

                obs[norm_idx, 7] = 1.0

        return obs.flatten()

    def _current_colour(self):
        return self.agent_colours[self.current_agent]

    def _pin(self, colour, pin_id):
        return next((pin for pin in self.boardPins if pin.color == colour and pin.id == pin_id), None)

    def _is_allowed_destination(self, colour, dest, pin_id=None, use_safe_rules=True):
        cell_type = self.board.cells[dest].postype
        target_colour = self.board.colour_opposites[colour]
        target_cells = set(self.board.axial_of_colour(target_colour))

        # -------------------------
        # Hard rules
        # -------------------------
        if pin_id is not None:
            pin = self._pin(colour, pin_id)
            if pin is None:
                return False

            # Once inside target, do not leave target
            if pin.axialindex in target_cells and dest not in target_cells:
                return False

        # Only board, own start, or own target are allowed
        if not (
            cell_type == "board"
            or cell_type == colour
            or cell_type == target_colour
        ):
            return False

        # -------------------------
        # Soft rules
        # -------------------------
        if use_safe_rules and pin_id is not None and self.recent_positions_per_pin > 0:
            recent_positions = self.pin_history.get((colour, pin_id), ())
            if dest in recent_positions:
                return False

        return True

    def legal_actions(self, colour=None, normalized=False):

        colour = colour or self._current_colour()
        key = (colour, normalized)

        if key in self._legal_cache:
            return self._legal_cache[key]

        hard_actions = []
        safe_actions = []

        start_hard_actions = []
        start_safe_actions = []

        source_cells = set(self.board.axial_of_colour(colour))

        for pin in self.boardPins:
            if pin.color != colour:
                continue

            is_start_piece = pin.axialindex in source_cells

            for dest in pin.getPossibleMoves():

                if self._is_allowed_destination(
                    colour,
                    dest,
                    pin.id,
                    use_safe_rules=False
                ):

                    action = (
                        self.encode_normalized_action(colour, pin.id, dest)
                        if normalized
                        else self.encode_action(pin.id, dest)
                    )

                    hard_actions.append(action)

                    if is_start_piece:
                        start_hard_actions.append(action)

                    if self._is_allowed_destination(
                        colour,
                        dest,
                        pin.id,
                        use_safe_rules=True
                    ):

                        safe_actions.append(action)

                        if is_start_piece:
                            start_safe_actions.append(action)

        actions = safe_actions if len(safe_actions) > 0 else hard_actions

        start_actions = (
            start_safe_actions
            if len(safe_actions) > 0
            else start_hard_actions
        )

        actions = self._prioritize_start_piece_actions(
            colour,
            actions,
            start_actions,
        )

        self._legal_cache[key] = actions

        return actions
    
    def _prioritize_start_piece_actions(
        self,
        colour,
        legal_actions,
        start_piece_actions,
    ):
        agent = next(
            agent for agent, agent_colour in self.agent_colours.items()
            if agent_colour == colour
        )

        turns = self.turns_per_player[agent]

        if turns > 25 and len(start_piece_actions) > 0:
            return start_piece_actions

        return legal_actions

    def action_mask(self, colour=None, normalized=False):
        colour = colour or self._current_colour()
        key = (colour, normalized)

        if key in self._mask_cache:
            return self._mask_cache[key]

        mask = np.zeros(self._action_space.n, dtype=np.float32)

        for action in self.legal_actions(colour, normalized=normalized):
            mask[action] = 1.0

        self._mask_cache[key] = mask
        return mask

    def action_masks(self):
        return self.action_mask(self._current_colour())

    def encode_action(self, pin_id, dest_idx):
        return int(pin_id) * self.num_cells + int(dest_idx)

    def decode_action(self, action_id):
        pin_id = int(action_id) // self.num_cells
        dest_idx = int(action_id) % self.num_cells
        return pin_id, dest_idx
    
    def encode_normalized_action(self, colour, pin_id, real_dest_idx):
        norm_dest_idx = self._norm_index(colour, real_dest_idx)
        return self.encode_action(pin_id, norm_dest_idx)


    def decode_normalized_action(self, colour, action_id):
        pin_id, norm_dest_idx = self.decode_action(action_id)
        real_dest_idx = self._real_index(colour, norm_dest_idx)
        return pin_id, real_dest_idx
    
    def step_normalized(self, actions):
        mover = self.current_agent
        colour = self.agent_colours[mover]

        norm_action = actions.get(mover, None)

        if norm_action is None:
            return self.step(actions)

        pin_id, real_dest = self.decode_normalized_action(colour, norm_action)
        real_action = self.encode_action(pin_id, real_dest)

        return self.step({mover: real_action})
    
    def min_dist_to_target(self, pin, target_cells):
        cell = self.board.cells[pin.axialindex]
        return min(
            hex_distance(
                cell.q,
                cell.r,
                self.board.cells[t].q,
                self.board.cells[t].r,
            )
            for t in target_cells
        )

    def _is_player_finished(self, colour):
        target_colour = self.board.colour_opposites[colour]
        target_cells = set(self.board.axial_of_colour(target_colour))
        positions = {p.axialindex for p in self.boardPins if p.color == colour}
        return positions.issubset(target_cells)

    def winner(self):
        for agent in self.agents:
            if self._is_player_finished(self.agent_colours[agent]):
                return agent
        return None

    def _advance_turn(self):
        current_index = self.agents.index(self.current_agent)
        self.current_agent = self.agents[(current_index + 1) % len(self.agents)]

    def step(self, actions):
        mover = self.current_agent
        colour = self.agent_colours[mover]
        action_id = actions.get(mover, None)

        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if action_id is None:
            rewards[mover] = 0.0
            infos[mover] = {"illegal": True, "reason": "missing_action", "player": mover}
            self.rewards = rewards
            self.infos = infos
            observations = {agent: self._observe(agent) for agent in self.agents}
            return observations, rewards, terminations, truncations, infos

        action_id = int(action_id)
        legal = set(self.legal_actions(colour))
        if action_id not in legal:
            rewards[mover] = 0.0
            infos[mover] = {"illegal": True, "reason": "action_not_legal", "player": mover}
            self.rewards = rewards
            self.infos = infos
            observations = {agent: self._observe(agent) for agent in self.agents}
            return observations, rewards, terminations, truncations, infos

        pin_id, dest = self.decode_action(action_id)
        pin = self._pin(colour, pin_id)
        if pin is None:
            rewards[mover] = 0.0
            infos[mover] = {"illegal": True, "reason": "pin_not_found", "player": mover}
            self.rewards = rewards
            self.infos = infos
            observations = {agent: self._observe(agent) for agent in self.agents}
            return observations, rewards, terminations, truncations, infos

        target_colour = self.board.colour_opposites[colour]
        target_cells = set(self.board.axial_of_colour(target_colour))

        old_pos = pin.axialindex

        own_start_cells = set(self.board.axial_of_colour(colour))

        old_pins_in_start = sum(
            1 for p in self.agent_pins[mover]
            if p.axialindex in own_start_cells
        )

        moved = pin.placePin(dest)
        self._clear_action_cache()

        if self.recent_positions_per_pin > 0:
            self.pin_history[(colour, pin.id)].append(dest)

        if not moved:
            rewards[mover] = 0.0
            infos[mover] = {"illegal": True, "reason": "placePin_failed", "player": mover}
            self.rewards = rewards
            self.infos = infos
            observations = {agent: self._observe(agent) for agent in self.agents}
            return observations, rewards, terminations, truncations, infos
        


        old_in_goal = old_pos in target_cells
        new_in_goal = dest in target_cells

        reward = -0.03 


        # Entered opponent goal
        if not old_in_goal and new_in_goal:
            reward += 1.0

        # Left opponent goal
        if old_in_goal and not new_in_goal:
            reward -= 1.0  

        new_goal_count = sum(
            1 for p in self.agent_pins[mover]
            if p.axialindex in target_cells
        )

        if new_goal_count >= 7 and not old_in_goal and new_in_goal:
            reward += 0.5

        new_pins_in_start = sum(
            1 for p in self.agent_pins[mover]
            if p.axialindex in own_start_cells
        )

        if new_pins_in_start < old_pins_in_start:
            reward += 0.3 * (old_pins_in_start - new_pins_in_start)

        rewards[mover] += reward

        self.num_turns += 1
        self.turns_per_player[mover] += 1

        done_by_turn_limit = self.turns_per_player[mover] >= self.max_turns_per_player
        winner = self.winner()
        done = winner is not None or done_by_turn_limit

        if done:
            if winner is not None:
                for agent in self.agents:
                    rewards[agent] += 20.0 if agent == winner else -5.0
                    terminations[agent] = True
            else:
                for agent in self.agents:
                            colour = self.agent_colours[agent]
                            target = set(self.board.axial_of_colour(self.board.colour_opposites[colour]))
                            in_goal = sum(1 for p in self.agent_pins[agent] if p.axialindex in target)
                            missing = self.num_pins_per_player - in_goal

                            rewards[agent] += 0.3 * in_goal
                            rewards[agent] -= 0.8 * missing

                            truncations[agent] = True


        infos[mover] = {
            "illegal": False,
            "player_that_moved": mover,
            "winner": winner,
            "turns": self.num_turns,
            "turns_per_player": self.turns_per_player.copy(),
            "rewards": rewards,
            "old_pos": old_pos,
            "dest": dest,
        }

        if not done:
            self._advance_turn()

        self.rewards = rewards
        self.infos = infos
        observations = {agent: self._observe(agent) for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def render(self):
        pins = [pin for pins in self.agent_pins.values() for pin in pins]
        self.board.print_ascii(pins=pins, empty="·")

    def close(self):
        return None


def parallel_env(num_players=2, max_turns_per_player=200, recent_positions_per_pin=3):
    return CheckersParallelEnv(
        num_players=num_players,
        max_turns_per_player=max_turns_per_player,
        recent_positions_per_pin=recent_positions_per_pin,
    )
