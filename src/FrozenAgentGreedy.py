from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
import gymnasium as gym

class FrozenAgentGreedy:
    def __init__(
        self,
        env,
		epsilon: float,
        discount_factor: float = 0.95,
    ):
        """
        Args:
            epsilon: The initial epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        #almacenamos los valores iniciales
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self.env = env

        #incializo el agente con estos parametros
        self.initAgent()

    def get_action(self, env, state: tuple[int]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        
        return self.epsilon_greedy_policy(state)        

    def updateStep(self, state: tuple[int], action: int, reward: float, terminated: bool, next_state: tuple[int]):
        """Actualiza a nivel de step"""
        self.episode.append((state, action))
        self.result_sum += self.factor * reward
        self.factor *= self.discount_factor

    def updateEpisode(self):
        """Actualiza a nivel de episodio"""
        for (state, action) in self.episode:
            self.n_visits[state, action] += 1.0
            alpha = 1.0 / self.n_visits[state, action]
            self.Q[state, action] += alpha * (self.result_sum - self.Q[state, action])
        # Guardamos datos sobre la evolución
        self.stats += self.result_sum
        self.list_stats.append(self.stats/(self.numEpisodes+1))
        self.list_episodes.append(len(self.episode))
        self.numEpisodes += 1

    def initAgent(self):
        #inicializa el agente con la configuración inicial
        #parametros del epsilon greedy y su decaimiento
        self.epsilon = self._epsilon
        self.discount_factor = self._discount_factor
        
        # Matriz de valores  Q
        self.nA = self.env.action_space.n
        self.Q = np.zeros([self.env.observation_space.n, self.nA])

        # Número de visitas. Vamos a realizar la versión incremental.
        self.n_visits = np.zeros([self.env.observation_space.n, self.nA])
        
        #para hacer trackiong del episodio
        self.episode=[]
        self.result_sum=0.0
        self.factor=1.0

        # Para mostrar la evolución en el terminal y algún dato que mostrar
        self.stats = 0.0
        self.list_stats = [self.stats]
        self.list_episodes = [self.stats]
        
        #para hacer tracking del total de episodios
        self.numEpisodes=0

    def initEpisode(self):
        #Inicializa el agente con unnuevo episodio
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

class FrozenAgentMC1:
    def __init__(self, env, epsilon: float, discount_factor: float = 0.95):
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self.env = env
        self.initAgent()

    def initAgent(self):
        self.epsilon = self._epsilon
        self.discount_factor = self._discount_factor
        self.nA = self.env.action_space.n
        self.Q = np.zeros([self.env.observation_space.n, self.nA])
        self.returns = defaultdict(list)  # Almacena todas las recompensas por estado-acción
        self.policy = np.ones((self.env.observation_space.n, self.nA)) / self.nA  # Política inicial uniforme
        self.numEpisodes = 0

    def get_action(self, state: tuple[int]) -> int:
        """Selecciona una acción basada en la política epsilon-greedy."""
        return np.random.choice(np.arange(self.nA), p=self.policy[state])

    def generate_episode(self):
        """Genera un episodio siguiendo la política actual."""
        episode = []
        state, _ = self.env.reset()
        done = False
        
        while not done:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated
        
        return episode
    
    def updateEpisode(self, episode):
        """Actualiza los valores Q usando Monte Carlo de primera visita."""
        G = 0  # Retorno acumulado
        visited = set()  # Rastrea las primeras visitas
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.discount_factor * G + reward
            if (state, action) not in visited:  # Solo primera visita
                visited.add((state, action))
                self.returns[(state, action)].append(G)
                self.Q[state, action] = np.mean(self.returns[(state, action)])
        
        self.updatePolicy()
        self.numEpisodes += 1

    def updatePolicy(self):
        """Actualiza la política haciendo que sea más greedy con respecto a Q."""
        for state in range(self.env.observation_space.n):
            best_action = np.argmax(self.Q[state])
            self.policy[state] = self.epsilon / self.nA
            self.policy[state, best_action] += (1 - self.epsilon)

    def train(self, num_episodes):
        """Ejecuta episodios para entrenar al agente."""
        for _ in range(num_episodes):
            episode = self.generate_episode()
            self.updateEpisode(episode)

class FrozenAgentMC:
    def __init__(self, env, epsilon: float, discount_factor: float = 0.95):
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self.env = env
        self.initAgent()

    def get_action(self, env, state: tuple[int]) -> int:
        return self.epsilon_greedy_policy(state)

    def updateStep(self, state: tuple[int], action: int, reward: float, terminated: bool, next_state: tuple[int]):
        self.episode.append((state, action))
        self.result_sum += self.factor * reward
        self.factor *= self.discount_factor

    def updateEpisode(self):
        G = 0  # Retorno acumulado
        visited = set()
        
        for (state, action) in self.episode[::-1]:
            G = self.discount_factor * G + self.result_sum
            if (state, action) not in visited:
                visited.add((state, action))
                self.n_visits[state, action] += 1.0
                alpha = 1.0 / self.n_visits[state, action]
                self.Q[state, action] += alpha * (G - self.Q[state, action])
        
        self.stats += self.result_sum
        self.list_stats.append(self.stats/(self.numEpisodes+1))
        self.list_episodes.append(len(self.episode))
        self.numEpisodes += 1
        self.updatePolicy()

    def initAgent(self):
        self.epsilon = self._epsilon
        self.discount_factor = self._discount_factor
        self.nA = self.env.action_space.n
        self.Q = np.zeros([self.env.observation_space.n, self.nA])
        self.n_visits = np.zeros([self.env.observation_space.n, self.nA])
        self.policy = np.ones((self.env.observation_space.n, self.nA)) / self.nA
        self.episode = []
        self.result_sum = 0.0
        self.factor = 1.0
        self.stats = 0.0
        self.list_stats = [self.stats]
        self.list_episodes = [self.stats]
        self.numEpisodes = 0

    def initEpisode(self):
        self.episode = []
        self.result_sum = 0.0
        self.factor = 1.0

    def decay_epsilon(self):
        self.epsilon = min(1.0, 1000.0/(self.numEpisodes+1))

    def random_epsilon_greedy_policy(self, state):
        pi_A = np.ones(self.nA, dtype=float) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        pi_A[best_action] += (1.0 - self.epsilon)
        return pi_A

    def epsilon_greedy_policy(self, state):
        pi_A = self.random_epsilon_greedy_policy(state)
        return np.random.choice(np.arange(self.nA), p=pi_A)

    def updatePolicy(self):
        for state in range(self.env.observation_space.n):
            best_action = np.argmax(self.Q[state])
            self.policy[state] = self.epsilon / self.nA
            self.policy[state, best_action] += (1 - self.epsilon)

    def train(self, num_episodes):
        for _ in range(num_episodes):
            self.initEpisode()
            state, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.get_action(self.env, state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.updateStep(state, action, reward, terminated, next_state)
                state = next_state
                done = terminated or truncated
            
            self.updateEpisode()

# Uso del agente en un entorno de ejemplo
env = gym.make("FrozenLake-v1", is_slippery=False)
agent = FrozenAgentMC1(env, epsilon=0.4)
agent.train(5000)
print(agent.Q,agent.policy)
