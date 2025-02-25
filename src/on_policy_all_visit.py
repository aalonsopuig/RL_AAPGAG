# Author: Till Zemann
# License: MIT License

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym

import FrozenAgent

import random


semilla=100
random.seed(semilla)
np.random.seed(semilla)



'''

%%capture
#@title Instalamos gym
!pip install 'gym[box2d]==0.20.0'

## Instalación de algunos paquetes.
#!apt-get update
## Para usar gymnasium[box2d]
#!apt install swig
#!pip install gymnasium[box2d]

#@title Importamos librerias
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
'''


print("definimos el entorno:")
#@title Importamos el lago helado
name = 'FrozenLake-v1'
env4 = gym.make(name, is_slippery=False, map_name="4x4", render_mode="ansi") # No resbaladizo para entender mejor los resultados.
env8 = gym.make(name, is_slippery=False, map_name="8x8", render_mode="ansi") # No resbaladizo para entender mejor los resultados.

#env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

def on_policy_all_visit(agent, env, num_episodes=5000, epsilon=0.4, decay=False, discount_factor=1):
    for episode in tqdm(range(n_episodes)):
        state, info = env.reset(seed=semilla)
        done = False
#        print("========================INIT",state,info)
    
        #inicializo el episodio
        agent.initEpisode()
    
        # play one episode
        while not done:
            if decay:
                agent.decay_epsilon()
            action = agent.get_action(env, state)
            next_state, reward, terminated, truncated, info = env.step(action)
 #           print(action,next_state, reward, terminated, truncated, info)

            # update the agent
            agent.updateStep(state, action, reward, terminated, next_state)

            # update if the environment is done and the current state
            done = terminated or truncated
            state = next_state

        #después de acabar el episodio actualizo la Q y el epsilon
        agent.updateEpisode()
    return agent.Q, agent.list_stats


def plot(list_stats):
  # Creamos una lista de índices para el eje x
  indices = list(range(len(list_stats)))

  # Creamos el gráfico
  plt.figure(figsize=(6, 3))
  plt.plot(indices, list_stats)

  # Añadimos título y etiquetas
  plt.title('Proporción de recompensas')
  plt.xlabel('Episodio')
  plt.ylabel('Proporción')

  # Mostramos el gráfico
  plt.grid(True)
  plt.show()




semilla=100
random.seed(semilla)
np.random.seed(semilla)


# hyperparameters
learning_rate = 0.01
n_episodes = 50000
start_epsilon = 0.4
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
discount_factor = 1.0

agent4 = FrozenAgent.FrozenAgentGreedy(
    env=env4,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor,
)

Q, list_stats = on_policy_all_visit(agent4, env4, num_episodes=50000, epsilon=0.4, discount_factor=1)

plot(agent4.list_stats)
print(f"Máxima proporcion: {agent4.list_stats[-1]}")


LEFT, DOWN, RIGHT, UP = 0,1,2,3
print("Valores Q para cada estado:\n", agent4.Q)



LEFT, DOWN, RIGHT, UP = 0,1,2,3
pi, actions = agent4.pi_star_from_Q(env4, agent4.Q)

print("Política óptima obtenida\n", pi, f"\n Acciones {actions} \n Para el siguiente grid\n", env4.render())
print()




semilla=100
random.seed(semilla)
np.random.seed(semilla)



env = env8


# hyperparameters
learning_rate = 0.01
n_episodes = 50000
start_epsilon = 0.4
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
discount_factor = 1.0

agent8 = FrozenAgent.FrozenAgentGreedy(
    env=env8,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor,
)


Q, list_stats = on_policy_all_visit(agent8, env8, num_episodes=n_episodes, epsilon=0.4, decay=True, discount_factor=1)


plot(agent8.list_stats)
print(f"Máxima proporcion: {agent8.list_stats[-1]}")
     
LEFT, DOWN, RIGHT, UP = 0,1,2,3
print("Valores Q para cada estado:\n", agent8.Q)


LEFT, DOWN, RIGHT, UP = 0,1,2,3
pi, actions = agent8.pi_star_from_Q(env8, agent8.Q)

print("Política óptima obtenida\n", pi, f"\n Acciones {actions} \n Para el siguiente grid\n", env8.render())
print()

