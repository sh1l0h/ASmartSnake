import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List
from game_state import SnakeGame
from astar_agent import AStarAI
from q_agent import QRunnerAgent
from nn_agent import NNRunnerAgent


@dataclass
class AgentMetrics:
    name: str
    scores: List[int]
    steps: List[int]
    episodes_completed: int
    current_score: int
    current_steps: int
    steps_without_food: int
    avg_score: float
    max_score: int
    avg_steps: float
    success_rate: float
    efficiency: float


class ModernVisualizer:

    def __init__(self, agent_names: List[str], max_episodes: int):
        self.agent_names = agent_names
        self.max_episodes = max_episodes
        self.metrics = {
            name:
            AgentMetrics(
                name=name,
                scores=[],
                steps=[],
                episodes_completed=0,
                current_score=0,
                current_steps=0,
                steps_without_food=0,
                avg_score=0.0,
                max_score=0,
                avg_steps=0.0,
                success_rate=0.0,
                efficiency=0.0,
            )
            for name in agent_names
        }

        self.colors = {
            'A*': '#FF6B6B',
            'Q-Learning': '#4ECDC4',
            'Neural Net': '#45B7D1',
        }

        self.setup_modern_plots()

    def setup_modern_plots(self):
        plt.style.use('seaborn-v0_8-darkgrid')

        self.fig = plt.figure(figsize=(18, 12), facecolor='#1e1e1e')
        self.fig.patch.set_facecolor('#1e1e1e')

        self.fig.suptitle(
            'Snake AI Performance Dashboard',
            fontsize=20,
            fontweight='bold',
            color='white',
            y=0.95,
        )

        gs = self.fig.add_gridspec(
            3,
            4,
            hspace=0.35,
            wspace=0.3,
            left=0.06,
            right=0.96,
            top=0.88,
            bottom=0.08,
        )

        self.ax_main = self.fig.add_subplot(gs[:2, :2])
        self.setup_main_chart()

        self.ax_current = self.fig.add_subplot(gs[0, 2:])
        self.setup_current_status()

        self.ax_stats = self.fig.add_subplot(gs[1, 2:])
        self.setup_stats_panel()

        self.ax_efficiency = self.fig.add_subplot(gs[2, 0])
        self.ax_success = self.fig.add_subplot(gs[2, 1])
        self.ax_steps = self.fig.add_subplot(gs[2, 2:])

        self.setup_metric_charts()

    def setup_main_chart(self):
        self.ax_main.set_facecolor('#2a2a2a')
        self.ax_main.set_title(
            'Score Progress Over Episodes',
            fontsize=14,
            fontweight='bold',
            color='white',
            pad=20,
        )
        self.ax_main.set_xlabel('Episode', fontsize=12, color='#cccccc')
        self.ax_main.set_ylabel('Score', fontsize=12, color='#cccccc')
        self.ax_main.grid(True, alpha=0.2, color='white')
        self.ax_main.tick_params(colors='#cccccc')

    def setup_current_status(self):
        self.ax_current.set_facecolor('#2a2a2a')
        self.ax_current.set_title(
            'Current Episode Status',
            fontsize=14,
            fontweight='bold',
            color='white',
            pad=20,
        )
        self.ax_current.tick_params(colors='#cccccc')

    def setup_stats_panel(self):
        self.ax_stats.set_facecolor('#2a2a2a')
        self.ax_stats.set_title(
            'Performance Statistics',
            fontsize=14,
            fontweight='bold',
            color='white',
            pad=20,
        )
        self.ax_stats.axis('off')

    def setup_metric_charts(self):
        charts = [
            (self.ax_efficiency, 'Efficiency', 'Score/Step'),
            (self.ax_success, 'Success Rate', 'Success %'),
            (self.ax_steps, 'Average Steps', 'Steps per Episode'),
        ]

        for ax, title, ylabel in charts:
            ax.set_facecolor('#2a2a2a')
            ax.set_title(
                title,
                fontsize=12,
                fontweight='bold',
                color='white',
                pad=15,
            )
            ax.set_ylabel(ylabel, fontsize=10, color='#cccccc')
            ax.tick_params(colors='#cccccc', labelsize=9)
            ax.grid(True, alpha=0.2, color='white')

    def update_metrics(
        self,
        agent_name: str,
        score: int,
        steps: int,
        steps_without_food: int,
        episode_complete: bool = False,
    ):
        metrics = self.metrics[agent_name]

        if episode_complete:
            metrics.scores.append(score)
            metrics.steps.append(steps)
            metrics.episodes_completed += 1
            metrics.current_score = 0
            metrics.current_steps = 0
            metrics.steps_without_food = 0

            if metrics.scores:
                metrics.avg_score = np.mean(metrics.scores)
                metrics.max_score = max(metrics.scores)
                metrics.avg_steps = np.mean(metrics.steps)
                metrics.success_rate = (sum(1
                                            for s in metrics.scores if s > 0) /
                                        len(metrics.scores)) * 100
                metrics.efficiency = (metrics.avg_score / metrics.avg_steps
                                      if metrics.avg_steps > 0 else 0)
        else:
            metrics.current_score = score
            metrics.current_steps = steps
            metrics.steps_without_food = steps_without_food

    def update_plots(self):
        for ax in [
                self.ax_main, self.ax_current, self.ax_stats,
                self.ax_efficiency, self.ax_success, self.ax_steps
        ]:
            ax.clear()

        self.setup_main_chart()
        self.setup_current_status()
        self.setup_stats_panel()
        self.setup_metric_charts()

        self.plot_main_performance()
        self.plot_current_status()
        self.plot_statistics()
        self.plot_metrics()

    def plot_main_performance(self):
        has_data = False

        for name, metrics in self.metrics.items():
            if metrics.scores and len(metrics.scores) > 0:
                has_data = True
                color = self.colors[name]
                episodes = range(1, len(metrics.scores) + 1)

                self.ax_main.plot(
                    episodes,
                    metrics.scores,
                    color=color,
                    alpha=0.6,
                    linewidth=1.5,
                    label=f'{name} (Raw)',
                    marker='o',
                    markersize=3,
                )

                if len(metrics.scores) >= 5:
                    window = min(10, len(metrics.scores))
                    moving_avg = np.convolve(
                        metrics.scores,
                        np.ones(window) / window,
                        mode='valid',
                    )
                    avg_episodes = range(window, len(metrics.scores) + 1)
                    self.ax_main.plot(
                        avg_episodes,
                        moving_avg,
                        color=color,
                        linewidth=3,
                        alpha=0.9,
                        label=f'{name} (Avg)',
                    )

        if has_data:
            self.ax_main.legend(
                loc='upper left',
                framealpha=0.9,
                facecolor='#1e1e1e',
                edgecolor='white',
            )

        if has_data:
            all_scores = [
                score for metrics in self.metrics.values()
                for score in metrics.scores
            ]
            if all_scores:
                self.ax_main.set_ylim(0, max(all_scores) * 1.1)

    def plot_current_status(self):
        agent_names = list(self.metrics.keys())
        current_scores = [
            self.metrics[name].current_score for name in agent_names
        ]
        steps_without_food = [
            self.metrics[name].steps_without_food for name in agent_names
        ]
        episodes_done = [
            self.metrics[name].episodes_completed for name in agent_names
        ]

        x_pos = np.arange(len(agent_names))
        colors = [self.colors[name] for name in agent_names]

        bars = self.ax_current.bar(
            x_pos,
            current_scores,
            color=colors,
            alpha=0.8,
            edgecolor='white',
            linewidth=1,
        )

        for i, (score, steps, eps) in enumerate(
                zip(current_scores, steps_without_food, episodes_done)):
            if score > 0:
                self.ax_current.text(
                    i,
                    score + 0.5,
                    f'{score}',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    color='white',
                    fontsize=11,
                )

            status_text = f'Ep: {eps}\nW/O Food: {steps}'
            self.ax_current.text(
                i,
                -2,
                status_text,
                ha='center',
                va='top',
                fontsize=9,
                color='#cccccc',
            )

        self.ax_current.set_xticks(x_pos)
        self.ax_current.set_xticklabels(agent_names, fontsize=10)
        self.ax_current.set_ylim(
            -5,
            max(current_scores) * 1.3 if current_scores else 10)

    def plot_statistics(self):
        stats_data = []

        for name, metrics in self.metrics.items():
            if metrics.episodes_completed > 0:
                color = self.colors[name]
                stats_data.append({
                    'name': name,
                    'episodes': metrics.episodes_completed,
                    'avg_score': metrics.avg_score,
                    'max_score': metrics.max_score,
                    'success_rate': metrics.success_rate,
                    'efficiency': metrics.efficiency,
                    'color': color
                })

        if not stats_data:
            self.ax_stats.text(
                0.5,
                0.5,
                'No data yet...',
                transform=self.ax_stats.transAxes,
                ha='center',
                va='center',
                fontsize=14,
                color='#888888',
                style='italic',
            )
            return

        y_start = 0.9
        y_step = 0.25

        header = f"{'Agent':<12} {'Episodes':<8} {'Avg Score':<10} {'Max':<6} {'Success%':<8} {'Efficiency':<10}"
        self.ax_stats.text(
            0.05,
            y_start,
            header,
            transform=self.ax_stats.transAxes,
            fontsize=10,
            fontweight='bold',
            color='white',
            fontfamily='monospace',
        )

        line_y = y_start - 0.05
        self.ax_stats.plot(
            [0.05, 0.95],
            [line_y, line_y],
            color='white',
            alpha=0.3,
            transform=self.ax_stats.transAxes,
        )

        for i, data in enumerate(stats_data):
            y_pos = y_start - (i + 1) * y_step
            row_text = (
                f"{data['name']:<12} {data['episodes']:<8} ",
                f"{data['avg_score']:<10.2f} {data['max_score']:<6} ",
                f"{data['success_rate']:<8.1f} {data['efficiency']:<10.4f}",
            )

            self.ax_stats.text(
                0.05,
                y_pos,
                row_text,
                transform=self.ax_stats.transAxes,
                fontsize=10,
                color=data['color'],
                fontfamily='monospace',
                fontweight='bold',
            )

    def plot_metrics(self):
        agent_names = [
            name for name, metrics in self.metrics.items()
            if metrics.episodes_completed > 0
        ]

        if not agent_names:
            return

        colors = [self.colors[name] for name in agent_names]

        efficiencies = [
            self.metrics[name].efficiency * 1000 for name in agent_names
        ]
        bars1 = self.ax_efficiency.bar(
            agent_names,
            efficiencies,
            color=colors,
            alpha=0.8,
        )
        self.ax_efficiency.set_ylabel(
            'Score/Step (×1000)',
            fontsize=10,
            color='#cccccc',
        )

        for bar, eff in zip(bars1, efficiencies):
            height = bar.get_height()
            self.ax_efficiency.text(
                bar.get_x() + bar.get_width() / 2.,
                height + height * 0.02,
                f'{eff:.2f}',
                ha='center',
                va='bottom',
                fontsize=9,
                color='white',
                fontweight='bold',
            )

        success_rates = [
            self.metrics[name].success_rate for name in agent_names
        ]
        bars2 = self.ax_success.bar(
            agent_names,
            success_rates,
            color=colors,
            alpha=0.8,
        )
        self.ax_success.set_ylim(0, 100)

        for bar, rate in zip(bars2, success_rates):
            height = bar.get_height()
            self.ax_success.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 2,
                f'{rate:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9,
                color='white',
                fontweight='bold',
            )

        avg_steps = [self.metrics[name].avg_steps for name in agent_names]
        bars3 = self.ax_steps.bar(
            agent_names,
            avg_steps,
            color=colors,
            alpha=0.8,
        )

        for bar, steps in zip(bars3, avg_steps):
            height = bar.get_height()
            self.ax_steps.text(
                bar.get_x() + bar.get_width() / 2.,
                height + height * 0.02,
                f'{steps:.0f}',
                ha='center',
                va='bottom',
                fontsize=9,
                color='white',
                fontweight='bold',
            )

        for ax in [self.ax_efficiency, self.ax_success, self.ax_steps]:
            ax.tick_params(axis='x', rotation=0, labelsize=9)

    def save_final_report(self, filename: str = 'storage/compare/snake_ai_comparison.png'):
        self.update_plots()
        self.fig.savefig(
            filename,
            dpi=300,
            bbox_inches='tight',
            facecolor='#1e1e1e',
            edgecolor='none',
        )
        print(f"Final report saved as {filename}")


class AgentTester:

    def __init__(
        self,
        episodes_per_agent: int = 30,
        max_steps_without_food: int = 200,
    ):
        self.episodes_per_agent = episodes_per_agent
        self.max_steps_without_food = max_steps_without_food

        self.agents = {
            'A*': lambda: AStarAI(),
            'Q-Learning': lambda: QRunnerAgent(20, 20),
            'Neural Net': lambda: NNRunnerAgent(20, 20)
        }

        self.visualizer = ModernVisualizer(
            list(self.agents.keys()),
            episodes_per_agent,
        )

        plt.ion()

    def test_agent(self, agent_factory, agent_name: str):
        print(
            f"\nTesting {agent_name} for {self.episodes_per_agent} episodes..."
        )

        for episode in range(self.episodes_per_agent):
            game = SnakeGame(20, 20)
            agent = agent_factory()
            steps = 0
            steps_without_food = 0
            last_score = 0

            while True:
                move = agent.next_move(game)
                status = getattr(game, f"move_{move}")()
                steps += 1

                current_score = len(game.snake) - 1

                if current_score > last_score:
                    steps_without_food = 0
                    last_score = current_score
                else:
                    steps_without_food += 1

                if steps % 20 == 0:
                    self.visualizer.update_metrics(
                        agent_name,
                        current_score,
                        steps,
                        steps_without_food,
                    )
                    self.visualizer.update_plots()
                    plt.pause(0.01)

                if (status == 'game_over'
                        or steps_without_food > self.max_steps_without_food):
                    break

            final_score = len(game.snake) - 1

            self.visualizer.update_metrics(
                agent_name,
                final_score,
                steps,
                steps_without_food,
                episode_complete=True,
            )

            self.visualizer.update_plots()
            plt.pause(0.01)

            metrics = self.visualizer.metrics[agent_name]
            print(
                f"  Episode {episode + 1:2d}/{self.episodes_per_agent}: ",
                f"Score={final_score:2d}, Steps={steps:3d}, ",
                f"Avg={metrics.avg_score:.1f}",
            )

        metrics = self.visualizer.metrics[agent_name]
        print(f"\n{agent_name} Results:")
        print(f"   Average Score: {metrics.avg_score:.2f}")
        print(f"   Max Score: {metrics.max_score}")
        print(f"   Success Rate: {metrics.success_rate:.1f}%")
        print(f"   Efficiency: {metrics.efficiency:.4f}")
        print("─" * 50)

    def run_comparison(self):
        print("Snake AI Agent Comparison Dashboard")
        print(f"Episodes per agent: {self.episodes_per_agent}")
        print(f"Max steps without food: {self.max_steps_without_food}")
        print("═" * 60)

        try:
            for agent_name, agent_factory in self.agents.items():
                self.test_agent(agent_factory, agent_name)

            self.visualizer.update_plots()
            self.visualizer.save_final_report()

            self.print_final_comparison()

            print("\nComparison complete! Close the plot window to exit.")

            plt.ioff()
            plt.show()

        except KeyboardInterrupt:
            print("\nComparison interrupted by user.")
        except Exception as e:
            print(f"Error during comparison: {e}")
        finally:
            plt.close('all')

    def print_final_comparison(self):
        print("\n" + "═" * 60)
        print("FINAL LEADERBOARD")
        print("═" * 60)

        sorted_agents = sorted(
            self.visualizer.metrics.items(),
            key=lambda x: x[1].avg_score,
            reverse=True,
        )

        for rank, (name, metrics) in enumerate(sorted_agents, 1):
            if metrics.episodes_completed > 0:
                medal = "[1st]" if rank == 1 else "[2nd]" if rank == 2 else "[3rd]"
                print(f"{medal} #{rank} {name}")
                print(f"    Average Score: {metrics.avg_score:.2f}")
                print(f"    Max Score: {metrics.max_score}")
                print(f"    Success Rate: {metrics.success_rate:.1f}%")
                print(f"    Efficiency: {metrics.efficiency:.4f}")
                print()

        if sorted_agents:
            winner = sorted_agents[0]
            print(
                f"Champion: {winner[0]} with {winner[1].avg_score:.2f} average score!"
            )


def main():
    EPISODES_PER_AGENT = 2
    MAX_STEPS_WITHOUT_FOOD = 200

    tester = AgentTester(
        episodes_per_agent=EPISODES_PER_AGENT,
        max_steps_without_food=MAX_STEPS_WITHOUT_FOOD,
    )

    tester.run_comparison()


if __name__ == '__main__':
    main()
