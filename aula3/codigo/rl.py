
import numpy as np


MAX_ESTADOS = 60
RECOMPENSA_SOBREVIVENCIA = 1.0
PENALIDADE_PERDEU = -99999.0
GAMMA = 0.99

class Agent:
    def __init__(self):
        self.policy = np.zeros((MAX_ESTADOS+1,MAX_ESTADOS+1))
    
        self.value_function = np.zeros((MAX_ESTADOS+1,MAX_ESTADOS+1))
    
    def nova_agua(self, nivel_agua, acao) -> int:
        return nivel_agua+acao
    
    def nova_temp(self, nivel_agua, nivel_temp) -> int:
        return nivel_temp+int(0.1*nivel_temp)-int(0.2*nivel_agua)
    
    def novo_valor(self, nivel_agua, nivel_temp, acao):
        proximo_agua_potencial = self.nova_agua(nivel_agua, acao)
        proximo_temp_potencial = self.nova_temp(nivel_agua, nivel_temp)
        
        perdeu = False
        valor_proximo_estado = 0 #V(s')

        #Temp negativa (nao esta nos estados validos, mas nn precisa de penalidade)
        if proximo_temp_potencial < 0:
            proximo_temp_potencial = max(0, proximo_temp_potencial)
        
        #Tentou saior do limite do nivel de agua
        if proximo_agua_potencial > MAX_ESTADOS or proximo_agua_potencial < 0:
            #Pode continuar
            proximo_agua_potencial = max(0, min(proximo_agua_potencial, MAX_ESTADOS))



        if proximo_temp_potencial >= MAX_ESTADOS:
            perdeu = True
        else:
            #Movimento valido
            valor_proximo_estado = self.value_function[int(proximo_agua_potencial)][int(proximo_temp_potencial)] #V(s')
        

        if(not perdeu):
            return (proximo_temp_potencial/MAX_ESTADOS)*(RECOMPENSA_SOBREVIVENCIA+GAMMA*valor_proximo_estado) + (1-(proximo_temp_potencial/MAX_ESTADOS))*(GAMMA*valor_proximo_estado)
        else:
            return PENALIDADE_PERDEU




    def policy_evaluation(self):

        while True:
            delta = 0

            #Loop por todos os estados
            for nivel_agua in range(MAX_ESTADOS+1):
                for nivel_temp in range(MAX_ESTADOS+1):
                    v_antigo = self.value_function[nivel_agua][nivel_temp]

                    #Perdeu
                    if nivel_temp >= MAX_ESTADOS:
                        self.value_function[nivel_agua][nivel_temp] = PENALIDADE_PERDEU
                        continue

                    self.value_function[nivel_agua][nivel_temp] = self.novo_valor(nivel_agua, nivel_temp, self.policy[nivel_agua][nivel_temp])
                    
                    delta = max(delta, abs(v_antigo - self.value_function[nivel_agua][nivel_temp]))

            # Checa a convergencia
            if delta < 0.0001:
                break

    def policy_improvement(self):
        policy_stable = True

        #Loop por todos os estados
        for nivel_agua in range(MAX_ESTADOS+1):
            for nivel_temp in range(MAX_ESTADOS+1):
                acao_antiga = self.policy[nivel_agua][nivel_temp]
                #Acoes possiveis
                acoes = [x for x in range(max(-(MAX_ESTADOS//10), -nivel_agua), min(MAX_ESTADOS//10, MAX_ESTADOS-nivel_agua))]
                valores = [self.novo_valor(nivel_agua, nivel_temp, acao) for acao in acoes]
                indx = valores.index(max(valores))
                self.policy[nivel_agua][nivel_temp] = acoes[indx]

                if(self.policy[nivel_agua][nivel_temp] != acao_antiga):
                    policy_stable = False
        
        return policy_stable