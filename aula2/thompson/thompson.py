### Créditos: github.com/petroud/E-greedy_and_th_algorithms/

import numpy as np
import matplotlib.pyplot as plt

# DEFINIÇÕES QUE MOLDAM O PROBLEMA NA MÁQUINA ------------------//

k = 10   # número de braços
T = 10000  # número de tentativas do agente

# Vetores do modelo
th_pulls = np.zeros(k)             # número de vezes que puxamos a alavanca
th_estimate_M = np.zeros(k)        # média de reward estimada p/alavanca 
th_total_rewards = np.zeros(k)     # total de reward para 1 alavanca
th_inst_score = np.zeros(T)        # reward do arm num tempo 't'
th_best_score = np.zeros(T)        # reward cumulativo em um arm no tempo 't1
th_alg_score = np.zeros(T)         # reward cumulativo no tempo 't'
th_regret = np.zeros(T)            # regret no tempo 't'
th_optimal_action = np.zeros(T)    # porcentagem de ações óptimas dadas até 't'

# Definindo nossas distribuições de probabilidade (uniformes p/ cada braço)
a = np.random.random(k)
b = np.random.random(k)
for i in range(k):
    if a[i] > b[i]:
        a[i], b[i] = b[i], a[i]

mean = (a + b) / 2   # contém a média de reward factual de cada alavanca

best = np.max(mean)  # média da melhor alavanca possível
index_best = np.where(mean == best)[0][0]  # index da alavanca c/melhor média 

# 'puxar' a alavanca (retorna recompensa)
def pull(i):
    return np.random.uniform(a[i], b[i])

success = 0
def success_rate(optimal, i, total):
    global success
    if i == index_best:   
        success += 1
    optimal[total] = success / (total+1)

# FUNÇÕES QUE DEFINEM O ALGORITMO DE THOMPSON ---------------------------//

def update_stats_th(reward, i, t):
    th_pulls[i] += 1 
    th_inst_score[t] = reward 
    th_total_rewards[i] += reward 
    th_best_score[t] = th_best_score[t-1] + best  
    th_alg_score[t] = th_alg_score[t-1] + th_inst_score[t] 
    th_estimate_M[i] = th_total_rewards[i] / th_pulls[i] 
    th_regret[t] = (th_best_score[t] - th_alg_score[t]) / (t+1)

# simulação
for t in range(1, T):
    # amostrando com base nos posteriors do modelo
    theta_samples = []
    for i in range(k):
        if (th_pulls[i] > 0):
            # posterior
            sample = np.random.normal(th_estimate_M[i], 1.0 / np.sqrt(th_pulls[i]))
        else:
            # inicializando alavanca nunca puxada
            sample = np.random.normal(0, 1.0)

        theta_samples.append(sample)

    chosen_bandit = np.argmax(theta_samples)
    
    reward = pull(chosen_bandit)
    update_stats_th(reward, chosen_bandit, t)
    success_rate(th_optimal_action, chosen_bandit, t)

# PLOTTAGEM --------------------------------------------------//

plt.title("Regret for T=" + str(T) + " rounds and k=" + str(k) + " bandits") 
plt.xlabel("Round T") 
plt.ylabel("Total Regret")
plt.plot(np.arange(1, T+1), th_regret, label='Thompson Sampling') 
plt.legend()
plt.show()

plt.title("Optimal action taken for T=" + str(T) + " rounds and k=" + str(k) + " bandits")
plt.xlabel("Round T") 
plt.ylabel("% Optimal action taken")
plt.plot(np.arange(1, T+1), th_optimal_action, label='Thompson Sampling') 
plt.legend()
plt.show()
