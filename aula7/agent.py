import numpy as np
from env import CartPoleEnv
import pygame

# Dado um estado (definido por posição, angulo, velocidade linear e velocidade angular), retorna o vetor de features dele
def get_polynomial_features(state: np.ndarray, degree: int = 2) -> np.ndarray:
    x, x_dot, theta, theta_dot = state
    x_norm = x / 2.4
    v_norm = np.clip(x_dot / 3.0, -1, 1)
    theta_norm = theta / 0.21
    omega_norm = np.clip(theta_dot / 2.0, -1, 1)
    
    if degree == 1:
        return np.array([1, x_norm, v_norm, theta_norm, omega_norm])
    
    elif degree == 2:
        return np.array([
            1,
            x_norm, v_norm, theta_norm, omega_norm,
            x_norm**2, v_norm**2, theta_norm**2, omega_norm**2,
            x_norm*v_norm, x_norm*theta_norm, x_norm*omega_norm,
            v_norm*theta_norm, v_norm*omega_norm, theta_norm*omega_norm
        ])
    
    else:
        raise ValueError("Only degree 1 or 2 supported")


class SemiGradientTDAgent:
    def __init__(self, num_features: int, num_actions: int, 
                 alpha: float = 0.01, gamma: float = 0.99):
        self.num_features = num_features
        self.num_actions = num_actions
        self.alpha = alpha

        # Fato de desconto
        self.gamma = gamma


        # Pesos da função valor: uma matriz (num_actions × num_features)
        # Cada linha representa os pesos para uma ação específica: Q(s,a) = w[a]^T · φ(s)
        # 
        # Por que precisamos de pesos separados para cada ação?
        # Porque do mesmo estado, ações diferentes podem ter valores completamente
        # diferentes. Por exemplo: se o poste está inclinado para direita, empurrar
        # para esquerda pode ser ruim (Q baixo) enquanto empurrar para direita pode
        # ser bom (Q alto). Com um único vetor de pesos, não conseguiríamos
        # representar essa diferença - teríamos Q(s,a1) = Q(s,a2) sempre.
        self.w = np.zeros((num_actions, num_features))
        
        self.num_updates = 0
    
    # Retonrna o valor de um par estado ação
    def get_q_value(self, features: np.ndarray, action: int) -> float:
        return np.dot(self.w[action], features)
    
    # Retorna o valor de todas as ações que você pode tomar em um dado estado
    def get_all_q_values(self, features: np.ndarray) -> np.ndarray:
        return np.array([self.get_q_value(features, a) for a in range(self.num_actions)])
    
    # Retorna uma ação de forma e-greedy
    def select_action(self, features: np.ndarray, epsilon: float = 0.1) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_values = self.get_all_q_values(features)
            # retorna a ação com melhor valor nesse estado
            return np.argmax(q_values)
    
    def update(self, features: np.ndarray, action: int, reward: float,
               next_features: np.ndarray, next_action: int, done: bool):
        current_q = self.get_q_value(features, action)
        
        if done:
            # se acabou, alvo é a recompensa observada
            target = reward 
        else:
            # se o episódio continua, o also é a recompensa mais o valor do próxima par estado-acao
            next_q = self.get_q_value(next_features, next_action)
            target = reward + self.gamma * next_q
        
        # Erro TD: (alvo - Q atual)
        td_error = target - current_q
        
        # Gradiente da função valor: como é linear, o gradiente é o vetor de features
        gradient = features
        
        # Atualizamos os pesos associado a ação
        # Isso é prático 
        self.w[action] += self.alpha * td_error * gradient
        
        self.num_updates += 1
        
        return td_error



if __name__ == "__main__":
    env = CartPoleEnv(render_mode="human")
    degree = 2
    num_features = 15
    num_actions = 2
    decay_factor = 0.995
    epsilon_start = 1
    
    agent = SemiGradientTDAgent(
        num_features=num_features,
        num_actions=num_actions,
        alpha=0.01,
        gamma=0.99
    )
    
    num_episodes = 1000
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        if(episode % 50 == 0):
            env.render_mode = "human"
        else :
            env.render_mode = None
        
        epsilon = epsilon_start * (decay_factor ** episode)
        
        # acha o vetor de features
        features = get_polynomial_features(state, degree=degree)
        # acha a ação e-greedy
        action = agent.select_action(features, epsilon=epsilon)
        
        while not done:
            # toma a ação, observa recompensa e próximo estado
            next_state, reward, done, _ = env.step(action)

            # Calcula as features do próximo estado
            next_features = get_polynomial_features(next_state, degree=degree)
            
            # Encontra a próxima ação e-greedy
            next_action = agent.select_action(next_features, epsilon=epsilon) if not done else 0
            
            # Atualiza os pesos da função valor utilizando SARSA
            agent.update(features, action, reward, next_features, next_action, done)
            
            state = next_state
            features = next_features
            action = next_action
            total_reward += reward
            
            if env.render_mode == "human":
                pygame.time.wait(10)
        
        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-5:])
        
        start_features = get_polynomial_features(np.zeros(4), degree=degree)
        q_values = agent.get_all_q_values(start_features)
        max_q = np.max(q_values)
        
        print(f"Episode {episode+1:2d} | Reward: {total_reward:3.0f} | "
              f"Avg(5): {avg_reward:5.1f} | Q(s₀): {max_q:6.1f} | ε: {epsilon}")
        
        if env.render_mode == "human":
            pygame.time.wait(200)
    
    print("-" * 60)
    print("\n📊 TRAINING SUMMARY")
    print(f"Total episodes: {num_episodes}")
    print(f"Total TD updates: {agent.num_updates}")
    print(f"Average reward (first 10): {np.mean(episode_rewards[:10]):.1f}")
    print(f"Average reward (last 10):  {np.mean(episode_rewards[-10:]):.1f}")
    print(f"Improvement: {np.mean(episode_rewards[-10:]) - np.mean(episode_rewards[:10]):.1f}")
    env.close()
