from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import gymnasium as gym
import FrozenAgent
import random
import time

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

ts1=[]
ts2=[]
ts3=[]

def setSemilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)

def train_agent(agent, env, num_episodes=5000, decay=False, semilla=1):
    agent.initAgent()
    for episode in tqdm(range(num_episodes)):
        state, info = env.reset(seed=semilla)
        done = False
    
        start_time=time.time_ns()
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
            
        t1 = time.time_ns()

        #después de acabar el episodio actualizo la Q y el epsilon
        agent.updateEpisode()
        
        t2 = time.time_ns()
        ts1.append((t2-start_time)/1000)
        ts2.append((t1-start_time)/1000)
        ts3.append((t2-t1)/1000)

def plot(agent):
  # Creamos una lista de índices para el eje x
  indices = list(range(len(agent.list_stats)))
  
  
  # Crear figura con dos subgráficos
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

  # Primer subplot
  ax1.plot(indices, agent.list_stats, label='stats')
  ax1.set_title('Proporción de recompensas')
  ax1.set_xlabel('Episodio')
  ax1.set_ylabel('Proporción')
  ax1.legend()
  ax1.grid(True)

  # Segundo subplot
  ax2.plot(indices, agent.list_episodes, label='episodios')
  ax2.set_title('Tamaño de episodios')
  ax2.set_xlabel('Episodio')
  ax2.set_ylabel('Tamaño')
  ax2.legend()
  ax2.grid(True)

  # Ajustar diseño y mostrar gráfico
  plt.tight_layout()
  plt.show()  



#inicializo los numeros aleatorios
setSemilla(semilla)

# hyperparameters
n_episodes = 50000
start_epsilon = 0.1
discount_factor = 0.99

#agent4 = FrozenAgent.FrozenAgentMC_On_All(
#agent4 = FrozenAgent.FrozenAgentMC_On_First(
#agent4 = FrozenAgent.FrozenAgentGreedy(
#agent4 = FrozenAgent.FrozenAgentMC_Off_Q(
#agent4 = FrozenAgent.FrozenAgentSARSA(
agent4 = FrozenAgent.FrozenAgentQ_Learning(
    env=env4,
    epsilon=start_epsilon,
    discount_factor=discount_factor,
    alpha=0.1
)

train_agent(agent4, env4, num_episodes=n_episodes, decay=False, semilla=semilla)



'''
agent4b = FrozenAgent.FrozenAgentMC_On_First(
    env=env4,
    epsilon=start_epsilon,
    discount_factor=discount_factor,
)

train_agent(agent4b, env4, num_episodes=n_episodes, decay=False, semilla=semilla)

np.set_printoptions(precision=6)
diff=np.array(agent4b.Q-agent4.Q)
print(type(diff))
print(f"Valores Q para cada estado:\n {agent4.Q}")
print(f"Valores Qb para cada estado:\n {agent4b.Q}")
print(f"Valores Q-Qb para cada estado:\n {diff}")
exit()
'''
plot(agent4)
print(f"Máxima proporcion: {agent4.list_stats[-1]}")

LEFT, DOWN, RIGHT, UP = 0,1,2,3
print("Valores Q para cada estado:\n", agent4.Q)
print("Valores pi para cada estado:\n", agent4.policy)

LEFT, DOWN, RIGHT, UP = 0,1,2,3
pi, actions = agent4.pi_star_from_Q(env4, agent4.Q)

print("Política óptima obtenida\n", pi, f"\n Acciones {actions} \n Para el siguiente grid\n", env4.render())
print()


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))

# Primer subplot
ax1.plot(np.arange(len(ts1)),ts1, label='ts1')
ax1.grid(True)
ax2.plot(np.arange(len(ts2)),ts2, label='ts2')
ax2.grid(True)
ax3.plot(np.arange(len(ts3)),ts3, label='ts3')
ax3.grid(True)
plt.tight_layout()
plt.show()  


exit()
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

train_agent(agent8, env8, num_episodes=n_episodes, decay=True, semilla=semilla)

plot(agent8)
print(f"Máxima proporcion: {agent8.list_stats[-1]}")
     
LEFT, DOWN, RIGHT, UP = 0,1,2,3
print("Valores Q para cada estado:\n", agent8.Q)

LEFT, DOWN, RIGHT, UP = 0,1,2,3
pi, actions = agent8.pi_star_from_Q(env8, agent8.Q)

print("Política óptima obtenida\n", pi, f"\n Acciones {actions} \n Para el siguiente grid\n", env8.render())
print()

