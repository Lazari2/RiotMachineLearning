import numpy as np

class Optimizer_Adam:
     # Otimizadores são tecnicas de ajuste de pesos durante o treinamente da rede neural
     # Otimizador Adam -> um dos mais utilizados em redes neurais profundas, utiliza medias moveis dos gradientes e magnitude
     # Adaptativo a parametros funciona bem com dados não estacionarios (não frequentes) e com dados estacionários

     def __init__(self, learning_rate = 0.001, decay=0, epsilon= 1e-7,
                  beta_1=0.9, beta_2=0.999):
          
          self.learning_rate = learning_rate 
          self.current_learning_rate = learning_rate
          self.decay = decay
          self.iterations = 0
          self.epsilon = epsilon
          self.beta_1 = beta_1
          self.beta_2 = beta_2

     def pre_update_params(self):
        if self.decay:
             self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    
     def update_params(self, layer):
          
          #se nao tiver nenhum array em cache ainda ele vai criar um novo preenchido com zeros:
          if not hasattr(layer, 'weight_cache'):
               layer.weight_momentums = np.zeros_like(layer.weights)
               layer.weight_cache = np.zeros_like(layer.weights)
               layer.bias_momentums = np.zeros_like(layer.biases)
               layer.bias_cache = np.zeros_like(layer.biases)
          
          layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
          layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

          weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
          bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

          #atualizar o cache com os novos gradientes
          layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
          layer.bias_cache = self.beta_2 * layer.bias_cache + (1 -self.beta_2) * layer.dbiases**2

          weight_cache_corrected = layer.weight_cache / (1 -self.beta_2 ** (self.iterations + 1))
          bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

          layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
          layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected)+ self.epsilon)

     def post_update_params (self):
          self.iterations += 1