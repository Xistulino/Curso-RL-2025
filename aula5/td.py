#/----------------CÓDIGO: WINDY-WORLD (livro do Sutton & Barto)-----------------------/
#
#   ALGORITMOS IMPLEMENTADOS: Q-Learning, Sarsa, Expected-Sarsa, Double
#   PARÂMETROS: livres para serem ajustados nos algoritmos
#   N° EPISÓDIOS: 300 (200 steps cada)
#
#   AMBIENTE:  grid 7x10; start = (3,0); end = (3,7)
#              Ações disponíveis: mover em toda direção (exceto em bordas)
#              Recompensas: -1 para todo movimento 
#              Vento: move o agente para cima sempre que ele entra em contato com ele
#              Colunas 3,4,5,8 tem Vento = 1; colunas 6,7 tem vento = 2
#
#/------------------------------------------------------------------------------------/

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math

#/------------------ CLASSE AMBIENTE -------------------------------------------------/
class WindyGridworld:
    ACTIONS = [( -1,  0),  # cima
               (  0,  1),  # direita
               (  1,  0),  # baixo
               (  0, -1)]  # esquerda
    
    def __init__(self, n_rows=7, n_cols=10, start=(3,0), goal=(3,7), max_steps=200):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.wind = [0,0,0,1,1,1,2,2,1,0][:n_cols]
        self.reset()
    
    # chamado quando o ambiente é criado; inicializa o agente (posição) e o n° 
    # de passos que ele deu
    def reset(self):
        self.agent = tuple(self.start)
        self.t = 0
        return self.agent
    
    # checa se player está dentro ou fora da grid
    def in_bounds(self, r, c):
        r = max(0, min(self.n_rows-1, r))
        c = max(0, min(self.n_cols-1, c))
        return (r, c)
    
    # recebe um passo (cima, baixo ...) e atualiza agente e ambiente de acordo
    def step(self, action):
        # atualizando row e column novas do agente
        dr, dc = WindyGridworld.ACTIONS[action] #ação = inteiro [0,3]
        r, c = self.agent
        new_r = r + dr
        new_c = c + dc
        # caso ele mova para lugares que não pode
        new_r, new_c = self.in_bounds(new_r, new_c)
        # aplicando vento para empurrar agente 
        wind_strength = self.wind[new_c]
        new_r = new_r - wind_strength
        new_r, new_c = self.in_bounds(new_r, new_c)
        self.agent = (new_r, new_c)
        self.t += 1
        # checando se atingiu objetivo OU já esgotou n° de passos 
        done = (self.agent == self.goal) or (self.t >= self.max_steps)
        reward = -1.0 if not (self.agent == self.goal) else 0.0  
        # retorna recompensa
        return self.agent, reward, done, {}

#/-------------FUNÇÕES AUXILIARES-----------------------------------------------------/ 
#política epsilon-gulosa: escolhe random OU com maior função ação-valor(Q)
def epsilon_greedy_action(Q, state, n_actions, epsilon):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    else:
        # pega Q pra cada ação disponível no estado atual
        qvals = Q[state]
        max_val = float(np.max(qvals))
        candidates = [a for a, v in enumerate(qvals) if v == max_val]
        # random.choice: caso haja empate
        return random.choice(candidates)

#volta um dicionário com tamanho 'n_actions' e valores inicializados de 0 p/Q
def make_Q(n_states, n_actions):
    return defaultdict(lambda: np.zeros(n_actions, dtype=float))

#pra poder jogar states no dicionário acima
def state_key(state):
    return tuple(state)

#calcula E(Q(s,a)) para o Expected Sarsa
def expected_value_under_epsilon_greedy(Q, next_state, n_actions, epsilon):
    q = Q[next_state]
    max_val = float(np.max(q))
    greedy_actions = [a for a, v in enumerate(q) if v == max_val]
    n_greedy = len(greedy_actions)
    prob = np.ones(n_actions) * (epsilon / n_actions)
    for a in greedy_actions:
        prob[a] += (1.0 - epsilon) / n_greedy  
    return np.dot(prob, q)

#/---------------ALGORITMOS DE RL-----------------------------------------------------/
def q_learning(env, num_episodes=200, alpha=0.5, gamma=1.0, epsilon=0.1, seed=None):
    #inicializando action-value functions p/todos os estados (tudo como 0)
    random.seed(seed); np.random.seed(seed)
    n_actions = len(env.ACTIONS)
    Q = make_Q(None, n_actions)
    returns = []
    steps_per_ep = []
    #loop externo: episódio a episódio (tudo é resetado)
    for ep in range(num_episodes):
        state = env.reset() 
        state = state_key(state)
        total_reward = 0.0
        done = False
        steps = 0
        #loop interno: passos do agente dentro do episódio
        while not done:
            #escolher ação da policy e-greedy co
            a = epsilon_greedy_action(Q, state, n_actions, epsilon)
            next_state, r, done, _ = env.step(a)
            next_state_key = state_key(next_state)
            total_reward += r
            steps += 1
            #atualizando Q com o target específico ('else' p/fim do episódio)
            #note o 'np.max': escolhe-se Q com o maior valor (política óptima)
            target = r + gamma * np.max(Q[next_state_key]) if not done else r
            Q[state][a] += alpha * (target - Q[state][a])
            state = next_state_key

        returns.append(total_reward)
        steps_per_ep.append(steps)

    return Q, returns, steps_per_ep

def sarsa(env, num_episodes=200, alpha=0.5, gamma=1.0, epsilon=0.1, seed=None):
    #inicializar tudo
    random.seed(seed); np.random.seed(seed)
    n_actions = len(env.ACTIONS)
    Q = make_Q(None, n_actions)
    returns = []
    steps_per_ep = []
    for ep in range(num_episodes):
        state = env.reset()
        state = state_key(state)
        a = epsilon_greedy_action(Q, state, n_actions, epsilon)
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            next_state, r, done, _ = env.step(a)
            next_state_key = state_key(next_state)
            total_reward += r
            steps += 1
            #if-else para caso seja estado terminal ou não
            if not done:
                a2 = epsilon_greedy_action(Q, next_state_key, n_actions, epsilon)
                target = r + gamma * Q[next_state_key][a2]
                #notem o target distinto
                Q[state][a] += alpha * (target - Q[state][a])
                state = next_state_key
                a = a2
            else:
                #caso seja um estado terminal, não use target, e sim o reward,p/atualizar
                #também não tome nenhuma ação nesse caso
                Q[state][a] += alpha * (r - Q[state][a])
                state = next_state_key

        returns.append(total_reward)
        steps_per_ep.append(steps)

    return Q, returns, steps_per_ep

#mesma coisa do Sarsa, com a diferença do target
def expected_sarsa(env, num_episodes=200, alpha=0.5, gamma=1.0, epsilon=0.1, seed=None):
    random.seed(seed); np.random.seed(seed)
    n_actions = len(env.ACTIONS)
    Q = make_Q(None, n_actions)
    returns = []
    steps_per_ep = []
    for ep in range(num_episodes):
        state = env.reset()
        state = state_key(state)
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            a = epsilon_greedy_action(Q, state, n_actions, epsilon)
            next_state, r, done, _ = env.step(a)
            next_state_key = state_key(next_state)
            total_reward += r
            steps += 1
            if not done:
                #target diferente
                expected_next = expected_value_under_epsilon_greedy(Q, next_state_key, n_actions, epsilon)
                target = r + gamma * expected_next
                Q[state][a] += alpha * (target - Q[state][a])
                state = next_state_key
            else:
                Q[state][a] += alpha * (r - Q[state][a])
                state = next_state_key
        returns.append(total_reward)
        steps_per_ep.append(steps)
    return Q, returns, steps_per_ep

def double_q_learning(env, num_episodes=200, alpha=0.5, gamma=1.0, epsilon=0.1, seed=None):
    random.seed(seed); np.random.seed(seed)
    n_actions = len(env.ACTIONS)
    #inicializando 2 action-value functions distintas!
    Q1 = make_Q(None, n_actions)
    Q2 = make_Q(None, n_actions)
    returns = []
    steps_per_ep = []
    for ep in range(num_episodes):
        state = env.reset()
        state = state_key(state)
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            #as funções ação-valor resultantes para a política gulosa analisar
            #serão a combinação de Q1 e Q2 (uma soma mesmo)
            combined_Q = defaultdict(lambda: np.zeros(n_actions, dtype=float))
            if random.random() < epsilon:
                a = random.randrange(n_actions)
            else:
                qsum = Q1[state] + Q2[state]
                max_val = float(np.max(qsum))
                candidates = [ai for ai, v in enumerate(qsum) if v == max_val]
                a = random.choice(candidates)
            next_state, r, done, _ = env.step(a)
            next_state_key = state_key(next_state)
            total_reward += r
            steps += 1
            #chance aleatória (50/50) de atualizar Q1 com Q2 ou o contrário
            if random.random() < 0.5:
                best_a = int(np.argmax(Q1[next_state_key]))
                target = r + gamma * Q2[next_state_key][best_a] if not done else r
                Q1[state][a] += alpha * (target - Q1[state][a])
            else:
                best_a = int(np.argmax(Q2[next_state_key]))
                target = r + gamma * Q1[next_state_key][best_a] if not done else r
                Q2[state][a] += alpha * (target - Q2[state][a])
            state = next_state_key
        returns.append(total_reward)
        steps_per_ep.append(steps)
    #Q final: Q1 + Q2
    Q = make_Q(None, n_actions)
    for s in list(set(list(Q1.keys()) + list(Q2.keys()))):
        Q[s] = Q1[s] + Q2[s]
    return Q, returns, steps_per_ep

#/---------PLOTTING & RUNNING (DESIMPORTANTE)-----------------------------------------/
def run_experiment(num_episodes=300, seed=0):
    env = WindyGridworld(n_rows=7, n_cols=10, start=(3,0), goal=(3,7), max_steps=200)
    alpha = 0.5
    gamma = 1.0
    epsilon = 0.1
    
    results = {}
    Q_q, returns_q, steps_q = q_learning(env, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
    Q_sarsa, returns_sarsa, steps_sarsa = sarsa(env, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
    Q_exp, returns_exp, steps_exp = expected_sarsa(env, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
    Q_double, returns_double, steps_double = double_q_learning(env, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
    
    results['q_learning'] = (Q_q, returns_q, steps_q)
    results['sarsa'] = (Q_sarsa, returns_sarsa, steps_sarsa)
    results['expected_sarsa'] = (Q_exp, returns_exp, steps_exp)
    results['double_q'] = (Q_double, returns_double, steps_double)

    return env, results

# função auxiliar p/ média móvel
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

env, results = run_experiment(num_episodes=300, seed=123)

plt.figure(figsize=(10,5))
episodes = np.arange(1, len(results['q_learning'][1]) + 1)
plt.plot(episodes, results['q_learning'][1], label='Q-Learning')
plt.plot(episodes, results['sarsa'][1], label='SARSA')
plt.plot(episodes, results['expected_sarsa'][1], label='Expected SARSA')
plt.plot(episodes, results['double_q'][1], label='Double Q')
plt.xlabel('Episódio')
plt.ylabel('Retorno')
plt.title('Curva de Aprendizado: Windy Gridworld')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(episodes, results['q_learning'][2], label='Q-Learning steps')
plt.plot(episodes, results['sarsa'][2], label='SARSA steps')
plt.plot(episodes, results['expected_sarsa'][2], label='Expected SARSA steps')
plt.plot(episodes, results['double_q'][2], label='Double Q steps')
plt.xlabel('Episódio')
plt.ylabel('Passos até objetivo')
plt.title('Passos por episódio')
plt.legend()
plt.grid(True)
plt.show()

window = 10
plt.figure(figsize=(10,5))
for name, (_, returns, _) in results.items():
    plt.plot(moving_average(returns, window), label=name)
plt.xlabel('Episódio')
plt.ylabel('Retorno (média móvel)')
plt.title(f'Retorno médio móvel (window={window})')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
for name, (_, _, steps) in results.items():
    plt.plot(moving_average(steps, window), label=name)
plt.xlabel('Episódio')
plt.ylabel('Passos até objetivo (média móvel)')
plt.title(f'Passos médios móveis por episódio (window={window})')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
for name, (_, returns, _) in results.items():
    success = [1 if r > -env.max_steps else 0 for r in returns]
    cum_success_rate = np.cumsum(success) / np.arange(1, len(success)+1)
    plt.plot(cum_success_rate, label=name)
plt.xlabel('Episódio')
plt.ylabel('Taxa de Sucesso Acumulada')
plt.title('Taxa de Sucesso ao longo do tempo')
plt.legend()
plt.grid(True)
plt.show()
