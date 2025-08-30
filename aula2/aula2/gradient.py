### Créditos: github.com/petroud/E-greedy_and_th_algorithms/

import numpy as np
import matplotlib.pyplot as plt

# DEFINIÇÕES QUE MOLDAM O PROBLEMA NA MÁQUINA ------------------//

k = 10# número de braços
T = 10000# número de tentativas do agente
alpha = 0.1# valor do step-size

# Vetores do modelo
grad_pulls = np.zeros(k) # número de vezes que puxamos a alavanca
grad_estimate_M = np.zeros(k)  # média de reward estimada p/alavanca 
grad_total_rewards = np.zeros(k) # total de reward para 1 alavanca
grad_inst_score = np.zeros(T) # reward do arm num tempo 't'
grad_best_score = np.zeros(T) # reward cumulativo em um arm no tempo 't1
grad_alg_score = np.zeros(T) # reward cumulativo no tempo 't'
grad_regret = np.zeros(T) # regret no tempo 't'
grad_optimal_action = np.zeros(T) # porcentagem de ações óptimas dadas até 't'

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

# FUNÇÕES QUE DEFINEM O ALGORITMO DO GRADIENTE ---------------------------//

# atualiza preferências do gradiente
def update_preferences(H, probability, chosen, reward, iterations):
    result = reward - np.sum(grad_total_rewards)/iterations

    for i in range(len(H)):
        # se é a que escolhemos
        if (i == chosen):
            H[i] += alpha * result * (1 - probability[i])
        # demais preferências
        else:
            H[i] -= alpha * result * probability[i]

def softmax(H):
    return (np.exp(H)/np.sum(np.exp(H)))

def update_stats_grad(reward, i, t):
    grad_pulls[i] += 1 
    grad_inst_score[t] = reward 
    grad_total_rewards[i] += reward 
    grad_best_score[t] = grad_best_score[t-1] + best  
    grad_alg_score[t] = grad_alg_score[t-1] + grad_inst_score[t] 
    grad_estimate_M[i] = grad_total_rewards[i] / grad_pulls[i] 
    grad_regret[t] = (grad_best_score[t] - grad_alg_score[t]) / (t+1)

#settando vetor de preferências para 0
H = np.zeros(k)

# simulação
for t in range(1, T):
    probability = softmax(H)
    # amostrando uma alavanca da distribuição dada pela softmax
    chosen_bandit = np.random.choice(np.arange(k), p=probability)
    reward = pull(chosen_bandit)

    update_preferences(H, probability, chosen_bandit, reward, t)
    update_stats_grad(reward, chosen_bandit, t)
    success_rate(grad_optimal_action, chosen_bandit, t)

# PLOTTAGEM --------------------------------------------------//

plt.title("Regret for T=" + str(T) + " rounds and k=" + str(k) + " bandits") 
plt.xlabel("Round T") 
plt.ylabel("Total Regret")
plt.plot(np.arange(1, T+1), grad_regret, label='Gradient Method') 
plt.legend()
plt.show()

plt.title("Optimal action taken for T=" + str(T) + " rounds and k=" + str(k) + " bandits")
plt.xlabel("Round T") 
plt.ylabel("% Optimal action taken")
plt.plot(np.arange(1, T+1), grad_optimal_action, label='Gradient Method') 
plt.legend()
plt.show()
