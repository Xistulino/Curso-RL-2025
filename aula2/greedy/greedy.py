### Créditos: https://github.com/petroud/E-greedy_and_UCB_algorithms/

import matplotlib.pyplot as plt
import numpy as np

#DEFINIÇÕES QUE MOLDAM O PROBLEMA NA MÁQUINA------------------//

k=10 # n° alavanca
T=10000 # n° tentativas

# Vetores do modelo
eg_pulls = np.zeros(k) # número de vezes que puxamos a alavanca
eg_estimate_M = np.zeros(k) # média de reward estimada p/alavanca 
eg_total_rewards = np.zeros(k) # total de reward para 1 alavanca
eg_inst_score = np.zeros(T) # reward do arm num tempo 't'
eg_best_score = np.zeros(T) # reward cumulativo em um arm no tempo 't1
eg_alg_score = np.zeros(T) # reward cumulativo no tempo 't'
eg_regret = np.zeros(T) # regret no tempo 't'
eg_optimal_action = np.zeros(T) # porcentagem de ações óptimas dadas até 't'

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

#FUNÇÕES QUE DEFINEM O ε-GREEDY--------------------------------//

def update_stats_epsilon(reward,i,t):
    # Atualizando vetores
    eg_pulls[i] += 1 
    eg_inst_score[t] = reward 
    eg_total_rewards[i] += reward 
    eg_best_score[t] = eg_best_score[t-1] + best 
    eg_alg_score[t] = eg_alg_score[t-1] + eg_inst_score[t]
    # Estimativa do agente da média de cada alavanca
    eg_estimate_M[i] = eg_total_rewards[i] / eg_pulls[i] 
    #Regret = escolha óptima - escolha feita
    eg_regret[t] = (eg_best_score[t] - eg_alg_score[t])/(t+1) 

# define a chance de exploração do nosso agente
eps=5.0 

# Simulação
for t in range(1,T):
    # Jogar um dado e checar se vamos ou não satisfazer 'eps'
    # (escolher uma alavanca aleatória)
    if np.random.rand() < eps: 
        # Random
        kth = np.random.randint(k)
        reward = pull(kth)
        update_stats_epsilon(reward, kth, t)
    else:
        # Alavanca + provável
        kth = np.argmax(eg_estimate_M) 
        reward = pull(kth) 
        update_stats_epsilon(reward, kth, t) 

    success_rate(eg_optimal_action, kth, t)
    
    # Reduzindo o valor de 'eps' conforme avançamos (explorar menos)
    eps = np.power(t,-1/3) * np.power(k*np.log(t), 1/3)

#PLOTTAGEM--------------------------------------------------//

plt.title("Regret for T=" +str(T)+ " rounds and k=" +str(k)+ " bandits") 
plt.xlabel("Round T") 
plt.ylabel("Total Regret")
plt.plot(np.arange(1,T+1),eg_regret, label='ε-greedy') 
plt.legend()
plt.show()

plt.title("Optimal action taken for T=" +str(T)+ " rounds and k=" +str(k)+ " bandits")
plt.xlabel("Round T") 
plt.ylabel("% Optimal action taken")
plt.plot(np.arange(1,T+1),eg_optimal_action, label='ε-greedy') 
plt.legend()
plt.show()
