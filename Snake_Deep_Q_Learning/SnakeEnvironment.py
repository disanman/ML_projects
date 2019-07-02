#                                     ____              _
#                                    / ___| _ __   __ _| | _____
#                                    \___ \| '_ \ / _` | |/ / _ \
#                                     ___) | | | | (_| |   <  __/
#                                    |____/|_| |_|\__,_|_|\_\___|
#                     _____            _                                      _
#                    | ____|_ ____   _(_)_ __ ___  _ __  _ __ ___   ___ _ __ | |_
#                    |  _| | '_ \ \ / / | '__/ _ \| '_ \| '_ ` _ \ / _ \ '_ \| __|
#                    | |___| | | \ V /| | | | (_) | | | | | | | | |  __/ | | | |_
#                    |_____|_| |_|\_/ |_|_|  \___/|_| |_|_| |_| |_|\___|_| |_|\__|
import numpy as np
from time import sleep


class SnakeEnv():

    def __init__(self, display=True):
        self.clock = 0.001
        self.display = display
        self.counter_start = 0
        self.rows = 10
        self.snake = np.array([[self.rows//2, self.rows//2]])   # half of the screen
        self.actions = ['up', 'right', 'down', 'left']
        self.direction = None  # snake has not moved yet
        self.action_movement = {'up': np.array([-1, 0]),
                                'right': np.array([0, 1]),
                                'down': np.array([1, 0]),
                                'left': np.array([0, -1])}
        self.rewards = {'closer': 0.3,
                        'farther': -0.3,
                        'grow': 10,
                        'dead': -10}
        self._random_food()
        # Execution variables
        self.episode = 0
        self.step = 0
        self.score = 0
        self.reward = 0
        self.cum_reward = 0
        self.dead = False
        self.crashed_body = False
        self.crashed_wall = False
        self.grow = False
        # Initialize state
        self.initial_state = self.get_state()
        self.new_state = None
        self.num_states = len(self.initial_state)

    def _random_food(self):
        while True:
            self.food = np.random.randint(low=0, high=self.rows, size=2)
            if not any(np.equal(self.snake, self.food).all(1)):
                break

    def get_state(self):
        self._update_pointers()
        self.state_distance_food_y, self.state_distance_food_x = self.food - self.snake_head
        # ------------------------------------------------
        # Vertical distances
        rows_of_body_elements_same_head_col = self.snake_body_rows[self.snake_body_cols == self.snake_head_col]
        # check up
        body_elements_up = rows_of_body_elements_same_head_col[rows_of_body_elements_same_head_col < self.snake_head_row]
        if len(body_elements_up) > 0:
            self.state_distance_up = self.snake_head_row - max(body_elements_up) - 1
        else:
            self.state_distance_up = self.snake_head_row
        # check down
        body_elements_down = rows_of_body_elements_same_head_col[rows_of_body_elements_same_head_col > self.snake_head_row]
        if len(body_elements_down) > 0:
            self.state_distance_down = min(body_elements_down) - self.snake_head_row - 1
        else:
            self.state_distance_down = self.rows - self.snake_head_row - 1
        # ------------------------------------------------
        # Horizontal distances
        cols_of_body_elements_same_head_row = self.snake_body_cols[self.snake_body_rows == self.snake_head_row]
        # check right
        body_elements_right = cols_of_body_elements_same_head_row[cols_of_body_elements_same_head_row > self.snake_head_col]
        if len(body_elements_right) > 0:
            self.state_distance_right = min(body_elements_right) - self.snake_head_col - 1
        else:
            self.state_distance_right = self.rows - self.snake_head_col - 1
        # check left
        body_elements_left = cols_of_body_elements_same_head_row[cols_of_body_elements_same_head_row < self.snake_head_col]
        if len(body_elements_left) > 0:
            self.state_distance_left = self.snake_head_row - max(body_elements_left) - 1
        else:
            self.state_distance_left = self.snake_head_col
        # ----------------------------------------------------
        # Snake length
        self.state_long_snake = len(self.snake) > 2
        # create state vector
        self.state = np.array((
            self.state_distance_food_y,
            self.state_distance_food_x,
            self.state_distance_up,
            self.state_distance_right,
            self.state_distance_down,
            self.state_distance_left,
            self.state_long_snake
            ))
        return self.state

    def _update_pointers(self):
        self.snake_head = self.snake[0]
        self.snake_head_row = self.snake_head[0]
        self.snake_head_col = self.snake_head[1]
        self.snake_body = self.snake[1:]
        self.snake_body_rows = self.snake_body[:, 0]
        self.snake_body_cols = self.snake_body[:, 1]
        self.food_row = self.food[0]
        self.food_col = self.food[1]
        self.snake_tail = self.snake_body[-1] if len(self.snake_body) > 0 else None

    def _update_screen(self):
        self.screen = np.zeros((self.rows, self.rows))
        # Set food to -2
        self.screen[self.food_row, self.food_col] = -2
        # Set snake head to -1
        self.screen[self.snake_head_row, self.snake_head_col] = -1
        # Set snake body to 1 (except the tail)
        self.screen[self.snake_body_rows[:-1], self.snake_body_cols[:-1]] = 1
        # Set snake tail to 2
        if self.snake_tail is not None:
            self.screen[self.snake_body_rows[-1], self.snake_body_cols[-1]] = 2

    def print_screen(self):
        self.get_state()
        if self.display:
            if not self.dead:
                self._update_screen()
                print('  ', ''.join(f'{x:2.0f}' for x in range(10)), end='\r\n')
                print(f'  +{"-"*2*(self.rows)}+', end='\r\n')
                for idx, row in enumerate(self.screen):
                    print(f'{idx:2.0f}|', end='')
                    for ch in row:
                        if ch == -2:  # food
                            print(' ●', end='')
                        elif ch == -1:  # snake's head
                            print(' ▣', end='')
                        elif ch == 1:  # snake's body
                            print(' ▪', end='')
                        elif ch == 2:  # snake's tail
                            print(' ·', end='')
                        else:
                            print('  ', end='')
                    print('|', end='\r\n')
                print(f'  +{"-"*2*(self.rows)}+', end='\r\n')
                print('\nState: ', ''.join(f'{x:3.0f}' for x in self.state))
                print('\nDistance food (y):', self.state_distance_food_y)
                print('Distance food (x):', self.state_distance_food_x)
                print('Distance up:      ', self.state_distance_up)
                print('Distance right:   ', self.state_distance_right)
                print('Distance down:    ', self.state_distance_down)
                print('Distance left:    ', self.state_distance_left)
            else:
                print('Crashed against own body') if self.crashed_body else None
                print('Crashed against wall') if self.crashed_wall else None
                print('Game lost!')
        print('\nReward: ', self.reward)
        print(f'Reward: {self.cum_reward:3.3f}', '(cum)')
        print('Score: ', self.score)

    def _calculate_reward(self):
        if self.grow:
            self.reward = self.rewards['grow']
        elif self.dead:
            self.reward = self.rewards['dead'] - 0.001 * self.step

        else:
            closer_to_food_y = (np.abs(self.new_state[0]) - np.abs(self.initial_state[0])) < 0
            closer_to_food_x = (np.abs(self.new_state[1]) - np.abs(self.initial_state[1])) < 0
            if closer_to_food_x | closer_to_food_y:
                self.reward = self.rewards['closer']

            else:
                self.reward = self.rewards['farther'] - 0.001 * self.step

        self.cum_reward += self.reward

    def move_snake(self, direction):
        assert direction in self.actions, 'Invalid direction'
        assert not self.dead, 'Snake has died, please restart the game'
        self.initial_state = self.get_state()
        self.direction = direction
        # Move the head of the snake:
        head_moved = self.snake_head + self.action_movement[direction]
        # grow if head arrived to food, else just move
        if (head_moved == self.food).all():
            self.grow = True
            self.snake = np.vstack((head_moved, self.snake))
            self._random_food()
            self.score += 1
        else:
            self.grow = False
            self.snake = np.vstack((head_moved, self.snake[:-1]))
        self.new_state = self.get_state()
        # Find if the snake died
        self.crashed_body = any(np.equal(self.snake_head, self.snake_body).all(1))
        self.crashed_wall = any(self.snake_head < 0) or any(self.snake_head >= self.rows)
        self.dead = self.crashed_body or self.crashed_wall
        self._calculate_reward()
        if self.display:
            self.print_screen()
            sleep(self.clock)
        return self.new_state, self.reward, self.dead

    def restart_game(self):
        self.__init__(display=self.display)
