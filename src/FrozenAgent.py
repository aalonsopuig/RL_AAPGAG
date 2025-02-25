from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym

class FrozenAgentBasic:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, env, state: tuple[int]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[state]))

    def update(
        self,
        state: tuple[int],
        action: int,
        reward: float,
        terminated: bool,
        next_state: tuple[int],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_state])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[state][action]
        )

        self.q_values[state][action] = (
            self.q_values[state][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class FrozenAgentGreedy:
    def __init__(
        self,
        env,
        learning_rate: float,
		initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        
        # Matriz de valores  Q
        self.nA = env.action_space.n
        self.Q = np.zeros([env.observation_space.n, self.nA])

        # Número de visitas. Vamos a realizar la versión incremental.
        self.n_visits = np.zeros([env.observation_space.n, self.nA])

        
        self.q_values = defaultdict(lambda: np.zeros(self.nA))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        #parametros del epsilon greedy y su decaimiento
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        
        #para hacer trackiong del episodio
        self.episode=[]
        self.result_sum=0.0
        self.factor=1.0

        # Para mostrar la evolución en el terminal y algún dato que mostrar
        self.stats = 0.0
        self.list_stats = [self.stats]
#        self.step_display = num_episodes / 10
		
        
        #para hacer tracking del total de episodios
        self.numEpisodes=0

    def get_action(self, env, state: tuple[int]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        
        return self.epsilon_greedy_policy(state)        

    def updateStep(
        self,
        state: tuple[int],
        action: int,
        reward: float,
        terminated: bool,
        next_state: tuple[int],
    ):
        """Updates the Q-value of an action."""
        self.episode.append((state, action))
        self.result_sum += self.factor * reward
        self.factor *= self.discount_factor
        self.training_error.append(self.factor * reward)

    def updateEpisode(self):
        for (state, action) in self.episode:
            self.n_visits[state, action] += 1.0
            alpha = 1.0 / self.n_visits[state, action]
            self.Q[state, action] += alpha * (self.result_sum - self.Q[state, action])
        # Guardamos datos sobre la evolución
        self.stats += self.result_sum
        self.list_stats.append(self.stats/(self.numEpisodes+1))
        self.numEpisodes += 1

    def initEpisode(self):
        self.episode=[]
        self.result_sum=0.0
        self.factor=1.0

    def decay_epsilon(self):
        #self.epsilon = min(self.final_epsilon, self.initial_epsilon/(self.numEpisodes+1))
        self.epsilon = min(1.0, 1000.0/(self.numEpisodes+1))


    # Política epsilon-soft. Se usa para el entrenamiento
    def random_epsilon_greedy_policy(self, state):
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    # Política epsilon-greedy a partir de una epsilon-soft
    def epsilon_greedy_policy(self, state):
        pi_A = self.random_epsilon_greedy_policy(state)
        return np.random.choice(np.arange(self.nA), p=pi_A)

    # Política Greedy a partir de los valones Q. Se usa para mostrar la solución.
    def pi_star_from_Q(self, env, Q):
        done = False
        pi_star = np.zeros([env.observation_space.n, env.action_space.n])
        state, info = env.reset() # start in top-left, = 0
        actions = ""
        while not done:
            action = np.argmax(Q[state, :])
            actions += f"{action}, "
            pi_star[state,action] = action
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return pi_star, actions

