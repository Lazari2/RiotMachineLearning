import numpy as np

class Activation_Sigmoid:
     #Mesmo princípio da softmax, transforma o output em probabilidade 
     #classificação binária : dois resultados possíveis - probabilidades de cada um
     #riot - probabilidade de ganhar / perder - talvez seja mais interessante utilizar a sigmoid do que o softmax
     
    def forward(self, inputs , training):
         
         self.inputs = inputs
         self.output = 1 / (1 + np.exp(-np.clip(inputs, -500, 500)))
    
    def backward (self, dvalues):
         
         self.dinputs = dvalues * (1 - self.output) * self.output
    
    def predictions(self, outputs):
         return (outputs>0.5)*1