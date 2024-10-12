import numpy as np

class Layer_Dropout:
#Essa classe desativa alguns neurônios aleatoriamente de acordo com a taxa de dropout, evita que o modelo dependa muito dos mesmos neurônios
#rate =  porcentual de neuronios que será desativado na "run"
     def __init__(self, rate):
          self.rate = 1 - rate

     def forward(self, inputs, training):
          self.inputs = inputs

          if not training:
               self.output = inputs.copy()
               return

          self.binary_mask = np.random.binomial(1, self.rate,
                                                size= inputs.shape) / self.rate

          self.output = inputs * self.binary_mask

     def backward(self, dvalues):
          self.dinputs = dvalues * self.binary_mask