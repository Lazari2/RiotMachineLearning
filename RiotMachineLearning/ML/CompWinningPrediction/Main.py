import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Class')))

from Class.Model import Model
from Class.Layer_Dense import Layer_Dense
from Class.Activation_ReLU import Activation_ReLU
from Class.Layer_Dropout import Layer_Dropout
from Class.Activation_Sigmoid import Activation_Sigmoid
from Class.Loss_BinaryCrossentropy import Loss_BinaryCrossentropy
from Class.Optimizer_Adam import Optimizer_Adam
from Class.Accuracy_Binary import Accuracy_Binary
from Class.Train import Train 
from Class.Saver import Saver
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_riot_dataset(path):
    data = pd.read_csv(path, delimiter=';')

    data = data.drop(columns=[
        'match_matchId', 'match_platformId', 'player_teamId', 
        'player_win', 'team_dragon_first', 'team_riftHerald_first'
    ])

    X = data.drop(columns=['team_winner']).values  
    y = data['team_winner'].values  

    y = y - 1

    return X, y

def create_riot_data(path):
    X, y = load_riot_dataset(path)

    # Embaralha os dados
    keys = np.arange(X.shape[0])
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test

path = r'C:\compPrediction\Project_Loud\RiotMachineLearning\ML\CompWinningPrediction\dataset\RiotDataSet3.csv'

X_train, y_train, X_test, y_test = create_riot_data(path)

print(f'Dados de treino: {X_train.shape}, {y_train.shape}')
print(f'Dados de teste: {X_test.shape}, {y_test.shape}')

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = Model()

model.add(Layer_Dense(50, 64, weight_regularizer_l2=1e-3))
model.add(Activation_ReLU())
#porcentagem de neuronios desativados a cada rotação
model.add(Layer_Dropout(0.4))
model.add(Layer_Dense(64, 32, weight_regularizer_l2=1e-3))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.4))
model.add(Layer_Dense(32, 1, weight_regularizer_l2=1e-3))
model.add(Activation_Sigmoid())  

model.set(
    loss=Loss_BinaryCrossentropy(),
    #valores altas decay-> faz previsões rápidas e focaliza
    #valores baixos decay-> ajuste mais agressivos no aprendizado por mais tempo
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Binary()
)

model.finalize()

train = Train(model)

history = train.train(
    X_train, y_train, validation_data=(X_test, y_test),
    epochs=10000, batch_size=256, print_every=100
)

train._evaluate_validation((X_test, y_test), batch_size=256)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(history['loss'], label='Training Loss')
ax1.plot(history['val_loss'], label='Validation Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(history['accuracy'], label='Training Accuracy')
ax2.plot(history['val_accuracy'], label='Validation Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout() 
plt.show()

saver = Saver()
saver.save_model(model, r'C:\compPrediction\Project_Loud\RiotMachineLearning\ML\CompWinningPrediction\models\model_riot7.pkl')

print("Modelo salvo com sucesso.")


