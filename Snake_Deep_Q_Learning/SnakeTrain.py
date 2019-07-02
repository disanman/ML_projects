from SnakeEnvironment import SnakeEnv
from SnakeDQNAgent import DQNAgent
import numpy as np
import click   # used to get input arguments when executed from a terminal cli


class SnakeTrainer():

    def __init__(self, train_episodes=5020, print_from=5000, steps_per_episode=200,
                 save_models=True, load_model='', exploration_rate=1):
        self.train_episodes = train_episodes
        self.print_from = print_from
        self.steps_per_episode = steps_per_episode
        self.display = steps_per_episode == 0
        # Initilize environment
        self.env = SnakeEnv(display=self.display)
        # Initialize agent
        self.agent = DQNAgent(self.env.num_states, self.env.actions)
        if load_model != '':
            self.agent.load_model(load_model)
        self.agent.exploration_rate = exploration_rate
        self.count = 0
        # Training variables
        self.state = self.env.state
        self.new_state = self.state
        self.save_models = save_models
        self.best_100_score = 0
        # Training memory
        self.episode_scores = [None] * train_episodes   # list pre-allocation for better memory management
        self.episode_cum_rewards = [None] * train_episodes

    def _train_episode(self, episode):
        for step in range(self.steps_per_episode):
            self.env.step = step
            self.state = self.env.state
            self.action_name = self.agent.act(self.state)
            self.new_state, self.reward, self.dead = self.env.move_snake(self.action_name)
            self.agent.remember(self.state, self.action_name, self.reward, self.new_state, self.dead)
            if self.display:
                print(f'\n\nEpisode: {episode}, step: {step}, exploration rate: {self.agent.exploration_rate}')
            if self.dead:
                break

    def _print_train_statistics(self, episode):
        if episode > 0:
            print(f'\nEpisodes ({episode-100}-{episode}):')
            scores = np.array(self.episode_scores[episode-100:episode])
            cum_rewards = np.array(self.episode_cum_rewards[episode-100:episode])
            print(f'  Mean scores:      {scores.mean():2.2f}, max: {scores.max():2.0f}')
            print(f'  Mean cum_rewards: {cum_rewards.mean():2.2f}, max: {cum_rewards.max():2.2f}')
            print(f'  Exploration rate: {self.agent.exploration_rate:2.3f}')
            print(f'  Learning rate:    {self.agent.alpha:2.3f}')
            print(f'  Discount rate:    {self.agent.discount_rate:2.3f}')
            if cum_rewards.mean() > self.best_100_score:
                self.agent.save_model(best=True)


    def train(self):
        for episode in range(self.train_episodes):
            self._train_episode(episode)
            self.episode_scores[episode] = self.env.score
            self.episode_cum_rewards[episode] = self.env.cum_reward
            self.env.restart_game()
            if episode >= self.print_from:
                self.display = self.env.display = True
                self.env.clock = 0.1
                self.agent.exploration_rate = 0
            if (episode % 50) == 0:
                self.agent.replay()
            if (episode % 100) == 0:  # Print training statistics each 100 epochs
                # Update exploration rate
                if self.agent.exploration_rate > self.agent.min_exploration_rate:
                    self.agent.exploration_rate *= self.agent.exploration_rate_decay
                # update target policy
                if self.agent.counter_to_update_target_nn >= self.agent.max_counter_to_update_target_nn:
                    self.agent.counter_to_update_target_nn = 0
                    self.agent._update_target_nn()
                else:
                    self.agent.counter_to_update_target_nn += 1
                # Update discount rate
                self.agent.discount_rate = min(0.5, self.agent.discount_rate + 0.001)  # so at 4000 it will be 1
                # ------------------
                self._print_train_statistics(episode)
            if self.save_models and (episode % 1000) == 0:   # save a model backup each 1000 episodes
                self.agent.save_model(episode)
                self.agent.alpha *= 0.99
                # self.agent.discount_rate += self.
        else:  # when finished all the episodes, save the last model
            self.agent.save_model(last=True)


@click.command()
@click.option('--train_episodes', default=5005, help='Number of episodes to train the snake agent')
@click.option('--print_from', default=5000, help='Minimum number of episodes to display the game screen')
@click.option('--steps_per_episode', default=200, help='Maximum number of steps per episode')
@click.option('--save_models', default=True, help='Whether or not to save models each 1000 episodes (Models folder)')
@click.option('--load_model', default='', help='Initial model to load')
@click.option('--exploration_rate', default=1.0, help='Exploration vs. exploitation rate')
def train_snake(train_episodes, print_from, steps_per_episode, save_models, load_model, exploration_rate):
    trainer = SnakeTrainer(train_episodes, print_from, steps_per_episode, save_models, load_model, exploration_rate)
    # trainer = SnakeTrainer()
    trainer.train()


if __name__ == '__main__':
    train_snake()
