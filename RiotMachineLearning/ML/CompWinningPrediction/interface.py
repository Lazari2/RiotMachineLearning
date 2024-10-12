import sys
import os
import numpy as np
import streamlit as st
from Class.Saver import Saver  
from Class.Train import Train

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Class')))

def get_team_vector(champion_list, all_champions):
    team_vector = np.zeros(50) 
    
    for champion in champion_list:
        if f"t1_{champion}" in all_champions:
            index = all_champions.index(f"t1_{champion}")
            team_vector[index] = 1  
        else:
            st.warning(f"Campeão '{champion}' não encontrado")
    
    return team_vector.reshape(1, -1)

model_path = r'C:\Data\Programmers.DataWarehouse\src\Riot Project\ApacheAirFlow\ML\CompWinningPrediction\models\model_riot7.pkl'

model = Saver.load_model(model_path) 

train = Train(model)

all_champions = [
    "t1_Aatrox","t1_Ashe", "t1_Alistar", "t1_Brand", "t1_Camille", "t1_Corki", "t1_Ezreal", "t1_Jhin", 
    "t1_Ksante", "t1_Kaisa", "t1_Lillia", "t1_Leona", "t1_LeBlanc", "t1_Nidalee", "t1_Nautilus", "t1_Renekton",
    "t1_Rumble", "t1_Tristana", "t1_LeeSin", "t1_Viego", "t1_Rell", "t1_Lucian", "t1_Yone", "t1_Zeri", "t1_Rakan"
]

st.title("Previsão de Probabilidade de Vitória - League of Legends")

st.write("Escolha os campeões do seu time :")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    champion1 = st.selectbox("TOP", ["Renekton", "Aatrox", "Ksante", "Rumble", "Camille"])

with col2:
    champion2 = st.selectbox("JUNGLE", ["LeeSin", "Nidalee", "Viego", "Lillia", "Brand"])

with col3:
    champion3 = st.selectbox("MID", ["Corki", "Tristana", "Lucian", "LeBlanc", "Yone"])

with col4:
    champion4 = st.selectbox("ADC", ["Kaisa", "Ezreal", "Zeri", "Jhin", "Ashe"])

with col5:
    champion5 = st.selectbox("SUPORTE", ["Leona", "Rell", "Nautilus", "Alistar", "Rakan"])

if st.button("Prever"):

    champions = [champion1, champion2, champion3, champion4, champion5]

    team_vector = get_team_vector(champions, all_champions)

    prediction = train.predict(team_vector)

    st.write(f"A chance de vitória para o time é de: {prediction[0][0] * 100:.2f}%")



#------------------------------------------------------------------------------------------------------------
#Campeões não utilizados por agora:

#     "t1_Diana", "t1_DrMundo", "t1_Draven", 
#     "t1_Ekko", "t1_Elise", "t1_Evelynn", "t1_Ezreal", "t1_FiddleSticks", "t1_Fiora", "t1_Fizz", 
#     "t1_Galio", "t1_Gangplank", "t1_Garen", "t1_Gnar", "t1_Gragas", "t1_Graves", "t1_Gwen", 
#     "t1_Hecarim", "t1_Heimerdinger", "t1_Hwei", "t1_Illaoi", "t1_Irelia", "t1_Ivern", "t1_Janna", 
#     "t1_JarvanIV", "t1_Jax", "t1_Jayce", "t1_Jhin", "t1_Jinx", "t1_KSante", "t1_Kaisa", 
#     "t1_Kalista", "t1_Karma", "t1_Karthus", "t1_Kassadin", "t1_Katarina", "t1_Kayle", "t1_Kayn", 
#     "t1_Kennen", "t1_Khazix", "t1_Kindred", "t1_Kled", "t1_KogMaw", "t1_LeBlanc", "t1_LeeSin", 
#     "t1_Leona", "t1_Lillia", "t1_Lissandra", "t1_Lucian", "t1_Lulu", "t1_Lux", "t1_Malphite", 
#     "t1_Malzahar", "t1_Maokai", "t1_MasterYi", "t1_MonkeyKing", "t1_Milio", "t1_MissFortune", 
#     "t1_Mordekaiser", "t1_Morgana", "t1_Naafiri", "t1_Nami", "t1_Nasus", "t1_Nautilus", 
#     "t1_Neeko", "t1_Nidalee", "t1_Nilah", "t1_Nocturne", "t1_Nunu", "t1_Olaf", "t1_Orianna", 
#     "t1_Ornn", "t1_Pantheon", "t1_Poppy", "t1_Pyke", "t1_Qiyana", "t1_Quinn", "t1_Rakan", 
#     "t1_Rammus", "t1_RekSai", "t1_Rell", "t1_Renata", "t1_Renekton", "t1_Rengar", "t1_Riven", 
#     "t1_Rumble", "t1_Ryze", "t1_Samira", "t1_Sejuani", "t1_Senna", "t1_Seraphine", "t1_Sett", 
#     "t1_Shaco", "t1_Shen", "t1_Shyvana", "t1_Singed", "t1_Sion", "t1_Sivir", "t1_Skarner", 
#     "t1_Smolder", "t1_Sona", "t1_Soraka", "t1_Swain", "t1_Sylas", "t1_Syndra", "t1_TahmKench", 
#     "t1_Taliyah", "t1_Talon", "t1_Taric", "t1_Teemo", "t1_Thresh", "t1_Tristana", "t1_Trundle", 
#     "t1_Tryndamere", "t1_TwistedFate", "t1_Twitch", "t1_Udyr", "t1_Urgot", "t1_Varus", 
#     "t1_Vayne", "t1_Veigar", "t1_Velkoz", "t1_Vex", "t1_Vi", "t1_Viego", "t1_Viktor", 
#     "t1_Vladimir", "t1_Volibear", "t1_Warwick", "t1_Xayah", "t1_Xerath", "t1_XinZhao", 
#     "t1_Yasuo", "t1_Yone", "t1_Yorick", "t1_Yuumi", "t1_Zac", "t1_Zed", "t1_Zeri", 
#     "t1_Ziggs", "t1_Zilean", "t1_Zoe", "t1_Zyra"