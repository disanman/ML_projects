from SnakeEnvironment import SnakeEnv
from SnakeDQNAgent import DQNAgent
import numpy as np

EPISODES = 6_050
MAX_STES = 200  # per episode
MIN_START_PRINT = 5000


def main():
    # Initialize
    env = SnakeEnv(display=False)
    agent = DQNAgent(env.num_states, env.actions)
    # Start
    episode_scores = []
    episode_cum_rewards = []
    count = 0
    for episode in range(EPISODES):
        for step in range(MAX_STES):
            state = env.state
            action_name = agent.act(state)
            new_state, reward, dead = env.move_snake(action_name)
            reward = reward
            agent.remember(state, action_name, reward, new_state, dead)
            print(f'\n\nEpisode: {episode}, step: {step}, exploration rate: {agent.exploration_rate}') if episode > MIN_START_PRINT else None
            if dead:
                break
        episode_scores.append(env.score)
        episode_cum_rewards.append(env.cum_reward)
        agent.replay()
        env.restart_game()
        env.display = True if episode > MIN_START_PRINT else False
        if (episode % 100) == 0:
            last_100_scores = np.array(episode_scores[-100:])
            last_100_cum_rewards = np.array(episode_cum_rewards[-100:])
            count += 100
            print(f'\n{count}: mean scores {last_100_scores.mean() / 100:.2f}, \
                    \nmean reward_cum {last_100_cum_rewards.mean() / 100:.2f} \
                    \nexploration rate: {agent.exploration_rate:.2f}')
        if (episode % 1000) == 0:
            agent.policy_nn.save(f'snake_model_{episode}.h5')

main()
