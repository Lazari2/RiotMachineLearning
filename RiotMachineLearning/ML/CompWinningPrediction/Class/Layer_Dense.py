import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        
        self.weights = 0.01* np.random.randn(n_inputs, n_neurons) # *0.01 para reduzir os pesos no inicio de treinamento, evitar explosões e/ou saturação de gradiente
        self.biases = np.zeros((1, n_neurons)) #inicializando bias com peso 0 -> modelo vai aprendendo gradualmente

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        self.inputs = inputs #insert inputs
        self.output = np.dot(inputs, self.weights) + self.biases # calculate the outputs
    
    #Calcular os gradientes no processo de backpropagation | ajustar o modelo com base nos erros de saídas
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #regularização dos pesos -> evitar overfitting
        if self.weight_regularizer_l1 > 0:
            #dL1 penalização proporcional ao gradiente dos pesos para ajustar a correção -> ajusta alguns pesos "forçando" a serem 0 eliminando overfitting
            #faz com que a rede neural exclua pesos irrelevantes e foque nos importantes
            #resumindo ajuste na proxima iteração da rede para melhorar o resulta
                dL1 = np.ones_like(self.weights)
                dL1[self.weights<0] = -1
                self.dweights += self.weight_regularizer_l1 * dL1 

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        if self.bias_regularizer_l1 > 0: 
             dL1 = np.ones_like(self.biases)
             dL1[self.biases <0] = -1
             self.dbiases += self.bias_regularizer_l1 * dL1
            
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        self.dinputs = np.dot(dvalues, self.weights.T)
    
    def get_parameters(self):
         return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
         self.weights = weights
         self.biases = biases