from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import gymnasium as gym
import FrozenAgent
import random


semilla=100



'''

%%capture
#@title Instalamos gym
!pip install 'gym[box2d]==0.20.0'

## Instalación de algunos paquetes.
#!apt-get update
## Para usar gymnasium[box2d]
#!apt install swig
#!pip install gymnasium[box2d]

'''


print("definimos el entorno:")
#@title Importamos el lago helado
name = 'FrozenLake-v1'
env4 = gym.make(name, is_slippery=False, map_name="4x4", render_mode="ansi") # No resbaladizo para entender mejor los resultados.
env8 = gym.make(name, is_slippery=False, map_name="8x8", render_mode="ansi") # No resbaladizo para entender mejor los resultados.

#env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)



def setSemilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)

def on_policy_all_visit(agent, env, num_episodes=5000, decay=False, semilla=1):
    agent.initAgent()
    for episode in tqdm(range(n_episodes)):
        state, info = env.reset(seed=semilla)
        done = False
    
        #inicializo el episodio
        agent.initEpisode()
    
        # play one episode
        while not done:
            if decay:
                agent.decay_epsilon()
            action = agent.get_action(env, state)
            next_state, reward, terminated, truncated, info = env.step(action)

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


#inicializo los numeros aleatorios
setSemilla(semilla)

# hyperparameters
n_episodes = 50000
start_epsilon = 0.4
discount_factor = 1.0

agent4 = FrozenAgent.FrozenAgentGreedy(
    env=env4,
    epsilon=start_epsilon,
    discount_factor=discount_factor,
)

Q, list_stats = on_policy_all_visit(agent4, env4, num_episodes=50000, decay=False, semilla=semilla)

plot(agent4.list_stats)
print(f"Máxima proporcion: {agent4.list_stats[-1]}")

LEFT, DOWN, RIGHT, UP = 0,1,2,3
print("Valores Q para cada estado:\n", agent4.Q)

LEFT, DOWN, RIGHT, UP = 0,1,2,3
pi, actions = agent4.pi_star_from_Q(env4, agent4.Q)

print("Política óptima obtenida\n", pi, f"\n Acciones {actions} \n Para el siguiente grid\n", env4.render())
print()


#ahora entrenamos con el mapa de 8x8

#inicializo los numeros aleatorios
setSemilla(semilla)

# hyperparameters
n_episodes = 50000
start_epsilon = 0.4
discount_factor = 1.0

agent8 = FrozenAgent.FrozenAgentGreedy(
    env=env8,
    epsilon=start_epsilon,
    discount_factor=discount_factor,
)

Q, list_stats = on_policy_all_visit(agent8, env8, num_episodes=n_episodes, decay=True, semilla=semilla)

plot(agent8.list_stats)
print(f"Máxima proporcion: {agent8.list_stats[-1]}")
     
LEFT, DOWN, RIGHT, UP = 0,1,2,3
print("Valores Q para cada estado:\n", agent8.Q)

LEFT, DOWN, RIGHT, UP = 0,1,2,3
pi, actions = agent8.pi_star_from_Q(env8, agent8.Q)

print("Política óptima obtenida\n", pi, f"\n Acciones {actions} \n Para el siguiente grid\n", env8.render())
print()

