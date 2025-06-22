from game_state import SnakeGame
from astar_agent import AStarAI
from q_agent import QLearningAI
from nn_agent import NeuralNetAI


def test_agent(agent_class, episodes=100):
    scores = []
    steps_list = []
    print(f"\nTesting {agent_class.__name__} for {episodes} games...\n")

    for ep in range(episodes):
        game = SnakeGame(20, 20)
        agent = agent_class()
        steps = 0

        while True:
            move = agent.next_move(game)
            status = getattr(game, f"move_{move}")()
            steps += 1
            if status == 'game_over':
                break

        score = len(game.snake) - 1
        scores.append(score)
        steps_list.append(steps)

        print(f"Episode {ep+1}: Score = {score}, Steps = {steps}")

    print(f"\nFinal results for {agent_class.__name__}:")
    print(f"  Avg Score: {sum(scores)/len(scores):.2f}")
    print(f"  Max Score: {max(scores)}")
    print(f"  Avg Steps: {sum(steps_list)/len(steps_list):.2f}")
    print("---------------")


def main():
    test_agent(AStarAI, episodes=50)
    test_agent(QLearningAI, episodes=50)
    test_agent(NeuralNetAI, episodes=50)


if __name__ == '__main__':
    main()
