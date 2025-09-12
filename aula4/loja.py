import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Tuple
import itertools

# Constantes do ambiente
ESTOQUE_MAX = 30
DEMANDA_MEDIA_A = 10   
DEMANDA_MEDIA_B = 5
PRECO_VENDA_A = 100         
PRECO_COMPRA_A = 80
PRECO_VENDA_B = 100
PRECO_COMPRA_B = 50
PENALIDADE_STOCKOUT = 20    # custo quando não há produto suficiente para atender a demanda
CUSTO_ESTOQUE = 1           # custo de armazenar estoque
GAMMA = 0.95                # fator de desconto do RL

class Loja:
    def nextState(self, state, action):
        stock_A, stock_B = state      # estado atual: estoque de A e B
        order_A, order_B = action     # ação: quantidade pedida de A e B

        # gera a demanda de A e B com base numa normal em torno da média
        demand_A = max(0, int(np.random.normal(DEMANDA_MEDIA_A, 0.5)))
        demand_B = max(0, int(np.random.normal(DEMANDA_MEDIA_B, 0.5)))
        
        # vendas limitadas ao que existe em estoque
        sales_A = min(stock_A, demand_A)
        sales_B = min(stock_B, demand_B)
        
        # falta de estoque (demanda não atendida)
        stockout_A = max(0, demand_A - stock_A)
        stockout_B = max(0, demand_B - stock_B)
        
        # contabilidade: receita, custos, penalidades
        revenue = sales_A * PRECO_VENDA_A + sales_B * PRECO_VENDA_B
        purchase_price = order_A * PRECO_COMPRA_A + order_B * PRECO_COMPRA_B
        penalty = (stockout_A + stockout_B) * PENALIDADE_STOCKOUT
        storage_cost = (stock_A + stock_B) * CUSTO_ESTOQUE
        
        reward = revenue - purchase_price - penalty - storage_cost  # lucro líquido
        
        # atualiza o estoque
        new_A = stock_A - sales_A + order_A
        new_B = stock_B - sales_B + order_B
        
        # garante que estoque não passe dos limites
        new_A = max(0, min(new_A, ESTOQUE_MAX))
        new_B = max(0, min(new_B, ESTOQUE_MAX))
        
        next_state = (new_A, new_B)
        return next_state, reward

class Agent:
    def __init__(self, env):
        self.env = env
        self.individual_actions = [0, 5, 10, 15, 20]  # ações possíveis para cada produto
        
        # combinações possíveis de ações (pedido de A e B)
        self.actions = list(itertools.product(self.individual_actions, self.individual_actions))
        self.numActions = len(self.actions)
        
        # política inicial: escolhe aleatoriamente uma ação pra cada estado
        self.policy = np.random.randint(0, self.numActions, (ESTOQUE_MAX + 1, ESTOQUE_MAX + 1))
        
        # inicializa a Q-table com valores aleatórios
        self.Q = np.random.random(size=(ESTOQUE_MAX + 1, ESTOQUE_MAX + 1, self.numActions))
        

    def get_policy_action(self, state):
        action_idx = self.policy[state]  # pega o índice da ação da política
        return self.actions[action_idx]

    def get_random_action(self):
        action_idx = np.random.choice(self.numActions)  # escolhe ação aleatória
        return self.actions[action_idx]

    def action_to_index(self, action):
        return self.actions.index(action)  # converte ação para índice

    def generateBehaviorEpisode(self, max_steps=100):
        # gera episódio seguindo política de comportamento (aleatória)
        episode = []
        state = (10, 10)  # estado inicial fixo

        for _ in range(max_steps):
            action = self.get_random_action()
            next_state, reward = self.env.nextState(state, action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def generatePolicyEpisode(self, max_steps=100):
        # gera episódio seguindo a política atual
        episode = []
        state = (10, 10)

        for _ in range(max_steps):
            action = self.get_policy_action(state)
            next_state, reward = self.env.nextState(state, action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def greedyPolicyState(self, state: Tuple[int, int]):
        # escolhe a ação gulosa (maior Q) no estado
        return np.argmax(self.Q[state])
    
    
    def MCControl(self, episodes: int = 1000):
        # Monte Carlo Control com importance sampling
        C = np.zeros((ESTOQUE_MAX + 1, ESTOQUE_MAX + 1, self.numActions))  # soma de pesos

        for episode_num in range(episodes):
            episode = self.generateBehaviorEpisode()
            
            G = 0  # retorno acumulado
            W = 1  # peso do importance sampling
            
            for (state, action, reward) in reversed(episode):
                G = GAMMA * G + reward  # atualiza retorno
                
                action_idx = self.action_to_index(action)
                
                # atualização de Q usando média ponderada
                C[state][action_idx] += W
                self.Q[state][action_idx] += (W / C[state][action_idx]) * (G - self.Q[state][action_idx])
                
                # atualiza a política para ser gulosa em relação a Q
                self.policy[state] = self.greedyPolicyState(state)
                
                # se a ação não for a mesma da política, interrompe
                if action_idx != self.policy[state]:
                    break
                
                # calcula peso do importance sampling
                behavior_prob = 1.0 / self.numActions  
                target_prob = 1.0 
                W *= (target_prob / behavior_prob)
