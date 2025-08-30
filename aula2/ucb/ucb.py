### Créditos: github.com/petroud/E-greedy_and_UCB_algorithms/

import matplotlib.pyplot as plt
import numpy as np

#DEFINIÇÕES QUE MOLDAM O PROBLEMA NA MÁQUINA------------------//

k = 10 # número de braços
T = 10000 # número de tentativas do agente

# Vetores do modelo
ucb_pulls = np.zeros(k) # número de vezes que puxamos a alavanca
ucb_estimate_M = np.zeros(k) # média de reward estimada p/alavanca 
ucb_total_rewards = np.zeros(k) # total de reward para 1 alavanca
ucb_inst_score = np.zeros(T) # reward do arm num tempo 't'
ucb_best_score = np.zeros(T) # reward cumulativo em um arm no tempo 't1
ucb_alg_score = np.zeros(T) # reward cumulativo no tempo 't'
ucb_regret = np.zeros(T) # regret no tempo 't'
ucb_optimal_action = np.zeros(T) # porcentagem de ações óptimas dadas até 't'

# Definindo nossas distribuições de probabilidade
a = np.random.random(k)
b = np.random.random(k)
for i in range(k):
    if(a[i] > b[i]):
        a[i],b[i] = b[i],a[i]

mean = (a+b)/2 # contém a média de reward factual de cada alavanca

best = np.max(mean) # média da melhor alavanca possível
index_best = np.where(mean==best)[0][0] # index da alavanca c/melhor média 

# 'puxar' a alavanca (retorna recompensa)
def pull(i):
    return np.random.uniform(a[i],b[i])

success = 0
# Acompanha a taxa de sucesso do modelo p/cada tentativa
def success_rate(optimal, i, total):
    global success
    if (i == index_best):
        success += 1

    optimal[total] = success/total

#FUNÇÕES QUE DEFINEM O ALGORITMO UCB---------------------------//

# Atualização do modelo de acordo com sua experiência
def update_stats_ucb(reward,i,t):
    # Atualizando nossos vetores da UCB
    ucb_pulls[i] += 1 
    ucb_inst_score[t] = reward 
    ucb_total_rewards[i] += reward 
    ucb_best_score[t] = ucb_best_score[t-1] + best  
    ucb_alg_score[t] = ucb_alg_score[t-1] + ucb_inst_score[t] 
    # Atualizando nossa estimativa sobre cada alavanca
    ucb_estimate_M[i] = ucb_total_rewards[i] / ucb_pulls[i] 
    # Regret pela decisão
    ucb_regret[t] = (ucb_best_score[t] - ucb_alg_score[t])/(t+1)


# simulação 
for t in range(1,T):
    # bonus por explorar (fórmula do UCB); o 0.0001 evita div/0
    exploration_bonus = np.sqrt(np.log(t)/(ucb_pulls + .0001))

    # Ciclo: UCB determina alavanca --> puxar alavanca --> agente se atualiza
    kth = np.argmax(ucb_estimate_M + exploration_bonus)
    reward = pull(kth)
    update_stats_ucb(reward, kth, t)

    success_rate(ucb_optimal_action, kth, t)

#PLOTTAGEM--------------------------------------------------//

plt.title("Regret for T=" +str(T)+ " rounds and k=" +str(k)+ " bandits") 
plt.xlabel("Round T") 
plt.ylabel("Total Regret")
plt.plot(np.arange(1,T+1),ucb_regret, label='UCB1') 
plt.legend()
plt.show()

plt.title("Optimal action taken for T=" +str(T)+ " rounds and k=" +str(k)+ " bandits")
plt.xlabel("Round T") 
plt.ylabel("% Optimal action taken")
plt.plot(np.arange(1,T+1),ucb_optimal_action, label='UCB1') 
plt.legend()
plt.show()
