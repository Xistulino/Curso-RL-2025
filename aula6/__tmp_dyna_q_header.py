import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from typing import Dict, Tuple, Set, Optional, Iterable

State = Tuple[int, int]
Action = int


class Maze:
    WALL = 1
    START = 2
    GOAL = 3

    def __init__(self) -> None:
        self.initial_layout = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=int)
        self.shortcut_layout = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 2, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=int)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self._set_layout(self.initial_layout)

    def _set_layout(self, layout: np.ndarray) -> None:
        self._layout = layout.copy()
        self.rows, self.cols = self._layout.shape
        self.start_state = self._locate(self.START)
        self.goal_state = self._locate(self.GOAL)

    def _locate(self, value: int) -> State:
        row, col = np.argwhere(self._layout == value)[0]
        return int(row), int(col)

    def reset(self) -> State:
        return self.start_state

    def layout(self) -> np.ndarray:
        return self._layout.copy()

    def _is_valid(self, row: int, col: int) -> bool:
        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
            and self._layout[row, col] != self.WALL
        )

    def step(self, state: State, action_idx: Action) -> Tuple[State, float, bool]:
        move = self.actions[action_idx]
        next_row = state[0] + move[0]
        next_col = state[1] + move[1]
        if not self._is_valid(next_row, next_col):
            next_state = state
        else:
            next_state = (next_row, next_col)
        reward = 1.0 if next_state == self.goal_state else 0.0
        done = next_state == self.goal_state
        return next_state, reward, done

    def open_shortcut(self) -> None:
        print("=== Shortcut opened in the maze ===")
        self._set_layout(self.shortcut_layout)


default_rng = np.random.default_rng


class DynaQAgent:
    def __init__(
        self,
        maze: Maze,
        n: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.0,
        optimistic_value: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.maze = maze
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimistic_value = optimistic_value
        self.num_actions = len(self.maze.actions)
        self.rng = default_rng(seed)
        self.q_table: Dict[State, np.ndarray] = defaultdict(self._init_q_values)
        self.model: Dict[Tuple[State, Action], Tuple[float, State]] = {}
        self.observed_pairs: Set[Tuple[State, Action]] = set()

    def _init_q_values(self) -> np.ndarray:
        return np.full(self.num_actions, self.optimistic_value, dtype=float)

    def _greedy_actions(self, state: State) -> np.ndarray:
        q_values = self.q_table[state]
        max_value = np.max(q_values)
        return np.flatnonzero(np.isclose(q_values, max_value))

    def choose_action(self, state: State) -> Action:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))
        candidates = self._greedy_actions(state)
        if len(candidates) == 1:
            return int(candidates[0])
        return int(self.rng.choice(candidates))

    def best_action(self, state: State) -> Action:
        return int(self._greedy_actions(state)[0])

    def _ensure_state_actions(self, state: State) -> Iterable[Tuple[State, Action]]:
        newly_added = []
        for action_idx in range(self.num_actions):
            key = (state, action_idx)
            if key not in self.model:
                self.model[key] = (0.0, state)
                self.observed_pairs.add(key)
                newly_added.append(key)
        return newly_added

    def update(self, state: State, action: Action, reward: float, next_state: State) -> None:
        next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        self.model[(state, action)] = (reward, next_state)
        self.observed_pairs.add((state, action))
        self._ensure_state_actions(state)
        self._ensure_state_actions(next_state)

    def planning(self) -> None:
        if not self.observed_pairs or self.n == 0:
            return
        observed = tuple(self.observed_pairs)
        for _ in range(self.n):
            state, action = observed[self.rng.integers(len(observed))]
            reward, next_state = self.model[(state, action)]
            next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * next_q
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * td_error


class DynaQPlusAgent(DynaQAgent):
    def __init__(
        self,
        maze: Maze,
        n: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.0,
        optimistic_value: float = 0.1,
        kappa: float = 1e-4,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            maze,
            n=n,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            optimistic_value=optimistic_value,
            seed=seed,
        )
        self.kappa = kappa
        self.current_step = 0
        self.last_visited: Dict[Tuple[State, Action], int] = defaultdict(int)

    def _ensure_state_actions(self, state: State) -> Iterable[Tuple[State, Action]]:
        newly_added = super()._ensure_state_actions(state)
        for pair in newly_added:
            self.last_visited[pair] = self.current_step
        return newly_added

    def update(self, state: State, action: Action, reward: float, next_state: State) -> None:
        self.current_step += 1
        super().update(state, action, reward, next_state)
        self.last_visited[(state, action)] = self.current_step

    def planning(self) -> None:
        if not self.observed_pairs or self.n == 0:
            return
        observed = tuple(self.observed_pairs)
        for _ in range(self.n):
            state, action = observed[self.rng.integers(len(observed))]
            reward, next_state = self.model[(state, action)]
            tau = self.current_step - self.last_visited[(state, action)]
            bonus = self.kappa * np.sqrt(max(tau, 0))
            adjusted_reward = reward + bonus
            next_q = np.max(self.q_table[next_state])
            td_target = adjusted_reward + self.gamma * next_q
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * td_error


# Remaining code stays the same as before...
