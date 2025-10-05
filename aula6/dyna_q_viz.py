import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from typing import Dict, Tuple, Set, Optional, Iterable, List

State = Tuple[int, int]
Action = int


def locate(layout: np.ndarray, value: int) -> State:
    row, col = np.argwhere(layout == value)[0]
    return int(row), int(col)


def is_valid(layout: np.ndarray, state: State) -> bool:
    rows, cols = layout.shape
    row, col = state
    return 0 <= row < rows and 0 <= col < cols and layout[row, col] != 1


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
        self.start_state = locate(self._layout, self.START)
        self.goal_state = locate(self._layout, self.GOAL)

    def reset(self) -> State:
        return self.start_state

    def layout(self) -> np.ndarray:
        return self._layout.copy()

    def step(self, state: State, action_idx: Action) -> Tuple[State, float, bool]:
        move = self.actions[action_idx]
        next_state = (state[0] + move[0], state[1] + move[1])
        if not is_valid(self._layout, next_state):
            next_state = state
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


def greedy_path(agent: DynaQAgent, layout: np.ndarray, max_steps: int = 200) -> List[State]:
    start = locate(layout, Maze.START)
    goal = locate(layout, Maze.GOAL)
    path = [start]
    state = start
    visited = {state}
    for _ in range(max_steps):
        if state == goal:
            break
        action = agent.best_action(state)
        move = agent.maze.actions[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        if not is_valid(layout, next_state):
            break
        path.append(next_state)
        if next_state in visited and next_state != goal:
            break
        visited.add(next_state)
        state = next_state
    return path


def plot_policies(layout: np.ndarray, agents: Dict[str, DynaQAgent], title: str) -> None:
    agent_items = list(agents.items())
    fig, axes = plt.subplots(1, len(agent_items), figsize=(6 * len(agent_items), 6))
    axes = np.atleast_1d(axes)
    fig.suptitle(title, fontsize=16)
    rows, cols = layout.shape
    arrow_map = {
        0: (0.0, -0.4),
        1: (0.0, 0.4),
        2: (-0.4, 0.0),
        3: (0.4, 0.0),
    }
    for ax, (name, agent) in zip(axes, agent_items):
        ax.set_title(name)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_aspect("equal")

        for row in range(rows):
            for col in range(cols):
                cell_value = layout[row, col]
                lower_left = (col - 0.5, row - 0.5)
                if cell_value == Maze.WALL:
                    ax.add_patch(patches.Rectangle(lower_left, 1, 1, facecolor="lightgray"))
                    continue
                ax.add_patch(patches.Rectangle(lower_left, 1, 1, facecolor="white", edgecolor="none"))
                if cell_value == Maze.START:
                    ax.add_patch(patches.Rectangle(lower_left, 1, 1, facecolor="cornflowerblue", alpha=0.6))
                    ax.text(col, row, "S", ha="center", va="center", color="white", fontweight="bold")
                elif cell_value == Maze.GOAL:
                    ax.add_patch(patches.Rectangle(lower_left, 1, 1, facecolor="seagreen", alpha=0.6))
                    ax.text(col, row, "G", ha="center", va="center", color="white", fontweight="bold")
                state = (row, col)
                if cell_value in (Maze.WALL, Maze.GOAL):
                    continue
                if state in agent.q_table:
                    action = agent.best_action(state)
                    dx, dy = arrow_map[action]
                    ax.arrow(
                        col,
                        row,
                        dx,
                        dy,
                        head_width=0.2,
                        head_length=0.2,
                        fc="black",
                        ec="black",
                        length_includes_head=True,
                    )
        path = greedy_path(agent, layout)
        if len(path) > 1:
            xs = [col for row, col in path]
            ys = [row for row, col in path]
            ax.plot(xs, ys, color="tab:red", linewidth=2, marker="o", markersize=4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def run_simulation(
    env: Maze,
    agents: Dict[str, DynaQAgent],
    total_steps: int,
    change_step: int,
    snapshot_step: int,
) -> None:
    states = {name: env.reset() for name in agents}
    initial_layout = env.layout()
    for step in range(1, total_steps + 1):
        if step == snapshot_step:
            print(f"Visualizing policies after {step - 1} steps (before shortcut).")
            plot_policies(
                initial_layout,
                agents,
                f"Policies after {step - 1} steps (before shortcut)",
            )
        if step == change_step:
            env.open_shortcut()
        for name, agent in agents.items():
            state = states[name]
            action = agent.choose_action(state)
            next_state, reward, done = env.step(state, action)
            agent.update(state, action, reward, next_state)
            agent.planning()
            states[name] = env.reset() if done else next_state
        if step % 1000 == 0:
            print(f"Simulation progress: {step}/{total_steps} steps")
    plot_policies(
        env.layout(),
        agents,
        f"Policies after {total_steps} steps (after shortcut)",
    )
    print("Simulation completed.")


def main() -> None:
    total_steps = 6000
    change_step = 3000
    snapshot_step = change_step - 1

    planning_steps = 25
    alpha = 0.5
    gamma = 0.95
    epsilon = 0.0
    optimistic_value = 0.1
    kappa = 1e-2
    seed = 0

    env = Maze()
    dyna_q = DynaQAgent(
        env,
        n=planning_steps,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        optimistic_value=optimistic_value,
        seed=seed,
    )
    dyna_q_plus = DynaQPlusAgent(
        env,
        n=planning_steps,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        optimistic_value=optimistic_value,
        kappa=kappa,
        seed=seed,
    )

    agents = {
        "Dyna-Q": dyna_q,
        "Dyna-Q+": dyna_q_plus,
    }

    run_simulation(env, agents, total_steps, change_step, snapshot_step)


if __name__ == "__main__":
    main()
