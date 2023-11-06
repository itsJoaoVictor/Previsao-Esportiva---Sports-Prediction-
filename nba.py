import pandas as pd
from sklearn import model_selection, preprocessing
import seaborn as sns
from scipy import stats
import sqlite3  
import json
import concurrent.futures
import requests
#Calcular o tempo de execução do script até este ponto
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

print("Fazendo a coleta dos dados e prevendo os resultados dos jogos da NBA...")


listnba21_22 = ['https://fixturedownload.com/feed/json/nba-2021/atlanta-hawks', 'https://fixturedownload.com/feed/json/nba-2021/boston-celtics','https://fixturedownload.com/feed/json/nba-2021/brooklyn-nets', 'https://fixturedownload.com/feed/json/nba-2021/charlotte-hornets', 'https://fixturedownload.com/feed/json/nba-2021/chicago-bulls', 'https://fixturedownload.com/feed/json/nba-2021/cleveland-cavaliers', 'https://fixturedownload.com/feed/json/nba-2021/dallas-mavericks', 'https://fixturedownload.com/feed/json/nba-2021/denver-nuggets', 'https://fixturedownload.com/feed/json/nba-2021/detroit-pistons', 'https://fixturedownload.com/feed/json/nba-2021/golden-state-warriors', 'https://fixturedownload.com/feed/json/nba-2021/houston-rockets', 'https://fixturedownload.com/feed/json/nba-2021/indiana-pacers', 'https://fixturedownload.com/feed/json/nba-2021/la-clippers', 'https://fixturedownload.com/feed/json/nba-2021/los-angeles-lakers', 'https://fixturedownload.com/feed/json/nba-2021/memphis-grizzlies', 'https://fixturedownload.com/feed/json/nba-2021/miami-heat', 'https://fixturedownload.com/feed/json/nba-2021/milwaukee-bucks', 'https://fixturedownload.com/feed/json/nba-2021/minnesota-timberwolves', 'https://fixturedownload.com/feed/json/nba-2021/new-orleans-pelicans', 'https://fixturedownload.com/feed/json/nba-2021/new-york-knicks', 'https://fixturedownload.com/feed/json/nba-2021/oklahoma-city-thunder', 'https://fixturedownload.com/feed/json/nba-2021/orlando-magic', 'https://fixturedownload.com/feed/json/nba-2021/philadelphia-76ers', 'https://fixturedownload.com/feed/json/nba-2021/phoenix-suns', 'https://fixturedownload.com/feed/json/nba-2021/portland-trail-blazers', 'https://fixturedownload.com/feed/json/nba-2021/sacramento-kings', 'https://fixturedownload.com/feed/json/nba-2021/san-antonio-spurs', 'https://fixturedownload.com/feed/json/nba-2021/toronto-raptors', 'https://fixturedownload.com/feed/json/nba-2021/utah-jazz', 'https://fixturedownload.com/feed/json/nba-2021/washington-wizards']


listnba22_23 = ['https://fixturedownload.com/feed/json/nba-2022/atlanta-hawks', 'https://fixturedownload.com/feed/json/nba-2022/boston-celtics','https://fixturedownload.com/feed/json/nba-2022/brooklyn-nets', 'https://fixturedownload.com/feed/json/nba-2022/charlotte-hornets', 'https://fixturedownload.com/feed/json/nba-2022/chicago-bulls', 'https://fixturedownload.com/feed/json/nba-2022/cleveland-cavaliers', 'https://fixturedownload.com/feed/json/nba-2022/dallas-mavericks', 'https://fixturedownload.com/feed/json/nba-2022/denver-nuggets', 'https://fixturedownload.com/feed/json/nba-2022/detroit-pistons', 'https://fixturedownload.com/feed/json/nba-2022/golden-state-warriors', 'https://fixturedownload.com/feed/json/nba-2022/houston-rockets', 'https://fixturedownload.com/feed/json/nba-2022/indiana-pacers', 'https://fixturedownload.com/feed/json/nba-2022/la-clippers', 'https://fixturedownload.com/feed/json/nba-2022/los-angeles-lakers', 'https://fixturedownload.com/feed/json/nba-2022/memphis-grizzlies', 'https://fixturedownload.com/feed/json/nba-2022/miami-heat', 'https://fixturedownload.com/feed/json/nba-2022/milwaukee-bucks', 'https://fixturedownload.com/feed/json/nba-2022/minnesota-timberwolves', 'https://fixturedownload.com/feed/json/nba-2022/new-orleans-pelicans', 'https://fixturedownload.com/feed/json/nba-2022/new-york-knicks', 'https://fixturedownload.com/feed/json/nba-2022/oklahoma-city-thunder', 'https://fixturedownload.com/feed/json/nba-2022/orlando-magic', 'https://fixturedownload.com/feed/json/nba-2022/philadelphia-76ers', 'https://fixturedownload.com/feed/json/nba-2022/phoenix-suns', 'https://fixturedownload.com/feed/json/nba-2022/portland-trail-blazers', 'https://fixturedownload.com/feed/json/nba-2022/sacramento-kings', 'https://fixturedownload.com/feed/json/nba-2022/san-antonio-spurs', 'https://fixturedownload.com/feed/json/nba-2022/toronto-raptors', 'https://fixturedownload.com/feed/json/nba-2022/utah-jazz', 'https://fixturedownload.com/feed/json/nba-2022/washington-wizards']

listnba23_24 = ['https://fixturedownload.com/feed/json/nba-2023/atlanta-hawks', 'https://fixturedownload.com/feed/json/nba-2023/boston-celtics','https://fixturedownload.com/feed/json/nba-2023/brooklyn-nets', 'https://fixturedownload.com/feed/json/nba-2023/charlotte-hornets', 'https://fixturedownload.com/feed/json/nba-2023/chicago-bulls', 'https://fixturedownload.com/feed/json/nba-2023/cleveland-cavaliers', 'https://fixturedownload.com/feed/json/nba-2023/dallas-mavericks', 'https://fixturedownload.com/feed/json/nba-2023/denver-nuggets', 'https://fixturedownload.com/feed/json/nba-2023/detroit-pistons', 'https://fixturedownload.com/feed/json/nba-2023/golden-state-warriors', 'https://fixturedownload.com/feed/json/nba-2023/houston-rockets', 'https://fixturedownload.com/feed/json/nba-2023/indiana-pacers', 'https://fixturedownload.com/feed/json/nba-2023/la-clippers', 'https://fixturedownload.com/feed/json/nba-2023/los-angeles-lakers', 'https://fixturedownload.com/feed/json/nba-2023/memphis-grizzlies', 'https://fixturedownload.com/feed/json/nba-2023/miami-heat', 'https://fixturedownload.com/feed/json/nba-2023/milwaukee-bucks', 'https://fixturedownload.com/feed/json/nba-2023/minnesota-timberwolves', 'https://fixturedownload.com/feed/json/nba-2023/new-orleans-pelicans', 'https://fixturedownload.com/feed/json/nba-2023/new-york-knicks', 'https://fixturedownload.com/feed/json/nba-2023/oklahoma-city-thunder', 'https://fixturedownload.com/feed/json/nba-2023/orlando-magic', 'https://fixturedownload.com/feed/json/nba-2023/philadelphia-76ers', 'https://fixturedownload.com/feed/json/nba-2023/phoenix-suns', 'https://fixturedownload.com/feed/json/nba-2023/portland-trail-blazers', 'https://fixturedownload.com/feed/json/nba-2023/sacramento-kings', 'https://fixturedownload.com/feed/json/nba-2023/san-antonio-spurs', 'https://fixturedownload.com/feed/json/nba-2023/toronto-raptors', 'https://fixturedownload.com/feed/json/nba-2023/utah-jazz', 'https://fixturedownload.com/feed/json/nba-2023/washington-wizards']

# Função para coletar dados da NFL para uma URL específica
def fetch_data(url):
    try:
        json_data = requests.get(url).json()
        return json_data
    except (requests.exceptions.RequestException, json.decoder.JSONDecodeError) as e:
        print(f"Erro na solicitação para {url}: {e}")
        return None

# Inicializar listas para armazenar os dados que você deseja extrair
match_numbers = []
round_numbers = []
dates_utc = []
home_teams = []
home_team_scores = []
away_teams = []
away_team_scores = []

# Combinação das listas de URLs
all_urls =  listnba21_22 + listnba22_23 + listnba23_24

# Use concurrent.futures para buscar dados assincronamente
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(fetch_data, all_urls)

# Iterar sobre os resultados e extrair os dados
for json_data in results:
    if json_data:
        for match in json_data:
            match_numbers.append(match.get('MatchNumber', None))
            round_numbers.append(match.get('RoundNumber', None))
            dates_utc.append(match.get('DateUtc', None))
            home_teams.append(match.get('HomeTeam', None))
            home_team_scores.append(match.get('HomeTeamScore', None))
            away_teams.append(match.get('AwayTeam', None))
            away_team_scores.append(match.get('AwayTeamScore', None))

# Agora você tem listas contendo os dados que precisa
# Você pode usá-los para criar um DataFrame do Pandas, que é útil para análise posterior

data = pd.DataFrame({
    'MatchNumber': match_numbers,
    'RoundNumber': round_numbers,
    'DateUtc': dates_utc,
    'HomeTeam': home_teams,
    'HomeTeamScore': home_team_scores,
    'AwayTeam': away_teams,
    'AwayTeamScore': away_team_scores
})

#Converter a coluna DateUtc para datetime
data['DateUtc'] = pd.to_datetime(data['DateUtc'])

#eliminar dados duplicados para ser duplicado todos os dados deve ser igual exceto o MatchNumber
data = data.drop_duplicates(subset=['RoundNumber','DateUtc','HomeTeam','HomeTeamScore','AwayTeam','AwayTeamScore'], keep='last')

#Ordenar os dados por data da mais antiga para a mais recente
data = data.sort_values(by=['DateUtc'])

#Filtrar os jogos realizados e os jogos que ainda serão realizados
jogos_realizados = data[data['HomeTeamScore'].notnull()]
jogos_future = data[data['HomeTeamScore'].isnull()]

#Analise dos dados
stats_jogos_realizados = jogos_realizados.describe()


#Pegar todos os times que jogaram na temporada
times = jogos_realizados['HomeTeam'].unique()

#Definir o sistema de Elo Rating para cada time inicialmente com 1500 pontos
elo = {}
for time in times:
    elo[time] = 1500
    
#Definir o K-Factor
K = 20

#Função para calcular a probabilidade de vitória de um time sobre o outro (Quero que a probabilidade seja algo real por exemplo 62.3% e não só numeros fechados como 0.5 ou 0.6)
def prob_vitoria(time1, time2):
    elo_diff = elo[time1] - elo[time2]
    return 1 / (10 ** (-elo_diff / 400) + 1)

#Função para atualizar o Elo Rating de um time após um jogo
def atualizar_elo(time1, time2, resultado):
    p1 = prob_vitoria(time1, time2)
    p2 = 1 - p1
    if resultado == 1:
        elo[time1] += K * (1 - p1)
        elo[time2] += K * (0 - p2)
    elif resultado == 0.5:
        elo[time1] += K * (0.5 - p1)
        elo[time2] += K * (0.5 - p2)
    else:
        elo[time1] += K * (0 - p1)
        elo[time2] += K * (1 - p2)
        
#Função para prever o resultado de um jogo
def prever_resultado(time1, time2):
    p1 = prob_vitoria(time1, time2)
    p2 = 1 - p1
    return p1, p2


for index, row in jogos_realizados.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    home_score = row['HomeTeamScore']
    away_score = row['AwayTeamScore']
    
    if home_score > away_score:
        resultado = 1  # Time da casa venceu
    elif home_score < away_score:
        resultado = 0  # Time visitante venceu
    else:
        resultado = 0.5  # Empate
    
    atualizar_elo(home_team, away_team, resultado)


#Criar um DataFrame com os dados de Elo Rating de cada time
elo_df = pd.DataFrame(elo.items(), columns=['Time', 'Elo Rating'])
elo_df = elo_df.sort_values(by=['Elo Rating'], ascending=False)
elo_df = elo_df.reset_index(drop=True)
elo_df.index += 1
elo_df.index.name = 'Posição'
#Arredondar o valor do Elo Rating para 2 casas decimais 
elo_df = elo_df.round({'Elo Rating': 2})


#prever a probabilidade de vitória de um time sobre o outro em um jogo futuro com base no Elo Rating atual
#Criar um DataFrame com os jogos futuros
jogos_future = jogos_future.reset_index(drop=True)
jogos_future.index += 1
jogos_future.index.name = 'Jogo'
jogos_future = jogos_future[['DateUtc','HomeTeam','AwayTeam']]
jogos_future['Casa%'] = 0.0
jogos_future['Fora%'] = 0.0
jogos_future['Elo Rating Time da Casa'] = 0.0
jogos_future['Elo Rating Time Visitante'] = 0.0

for index, row in jogos_future.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    p1, p2 = prever_resultado(home_team, away_team)
    jogos_future.loc[index, 'Casa%'] = p1
    jogos_future.loc[index, 'Fora%'] = p2
    jogos_future.loc[index, 'Elo Rating Time da Casa'] = elo[home_team]
    jogos_future.loc[index, 'Elo Rating Time Visitante'] = elo[away_team]
    
#Arredondar o valor da probabilidade de vitória para 2 casas decimais
jogos_future = jogos_future.round({'Casa%': 2, 'Fora%': 2})


#Criar um DataFrame com os jogos passados
jogos_realizados = jogos_realizados.reset_index(drop=True)
jogos_realizados.index += 1
jogos_realizados.index.name = 'Jogo'
jogos_realizados = jogos_realizados[['DateUtc','HomeTeam','AwayTeam','HomeTeamScore','AwayTeamScore']]
jogos_realizados['Elo Rating Time da Casa'] = 0.0
jogos_realizados['Elo Rating Time Visitante'] = 0.0

for index, row in jogos_realizados.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    jogos_realizados.loc[index, 'Elo Rating Time da Casa'] = elo[home_team]
    jogos_realizados.loc[index, 'Elo Rating Time Visitante'] = elo[away_team]
    

## Crie um DataFrame com os dados dos jogos realizados que serão usados para treinar o modelo
dados_treinamento = jogos_realizados[['HomeTeamScore', 'AwayTeamScore', 'Elo Rating Time da Casa', 'Elo Rating Time Visitante']]

# Separe as características (X) e o alvo (y)
X = dados_treinamento[['Elo Rating Time da Casa', 'Elo Rating Time Visitante']]
y_home = dados_treinamento['HomeTeamScore']
y_away = dados_treinamento['AwayTeamScore']

# Defina uma grade de hiperparâmetros para pesquisa
param_grid = {
    'n_estimators': [50, 100, 200],
    'random_state': [0, 42, 100]
}

# Crie o modelo de regressão para HomeTeamScore
modelo_home = RandomForestRegressor()

# Crie um objeto GridSearchCV para encontrar os melhores hiperparâmetros
grid_search_home = GridSearchCV(estimator=modelo_home, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Ajuste o modelo aos dados
grid_search_home.fit(X, y_home)

# Obtenha o melhor modelo com os melhores hiperparâmetros
melhor_modelo_home = grid_search_home.best_estimator_

# Crie o modelo de regressão para AwayTeamScore
modelo_away = RandomForestRegressor()

# Crie um objeto GridSearchCV para encontrar os melhores hiperparâmetros
grid_search_away = GridSearchCV(estimator=modelo_away, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Ajuste o modelo aos dados
grid_search_away.fit(X, y_away)

# Obtenha o melhor modelo com os melhores hiperparâmetros
melhor_modelo_away = grid_search_away.best_estimator_


# Agora, você pode usar os melhores modelos para fazer previsões nos jogos futuros
dados_previsao = jogos_future[['Elo Rating Time da Casa', 'Elo Rating Time Visitante']]
previsao_home = melhor_modelo_home.predict(dados_previsao)
previsao_away = melhor_modelo_away.predict(dados_previsao)

# Adicione as previsões ao DataFrame dos jogos futuros
jogos_future['xG Home'] = previsao_home
jogos_future['xG Away'] = previsao_away

# Arredonde os valores das previsões para inteiros, se necessário
jogos_future['xG Home'] = jogos_future['xG Home'].astype(int)
jogos_future['xG Away'] = jogos_future['xG Away'].astype(int)
#coluna para o total de pontos e handicap
jogos_future['Total Pontos'] = jogos_future['xG Home'] + jogos_future['xG Away']
jogos_future['Handicap'] = jogos_future['xG Home'] - jogos_future['xG Away']
#Odd das probabilidades de vitória
jogos_future['Odd Casa'] = 1 / jogos_future['Casa%']
jogos_future['Odd Fora'] = 1 / jogos_future['Fora%']
#Arredondar os valores das odds para 2 casas decimais
jogos_future = jogos_future.round({'Odd Casa': 2, 'Odd Fora': 2})
#Reordenar as colunas
jogos_future = jogos_future[['DateUtc','HomeTeam','AwayTeam','xG Home','xG Away','Total Pontos','Handicap','Casa%','Fora%','Odd Casa','Odd Fora']]

                          


#Armazenar os dados em um banco de dados
conn = sqlite3.connect('nba.db')
c = conn.cursor()
#O DateUtc está no +00:00 eu quero que esteja no -03:00 (Horário de Brasília)
jogos_realizados['DateUtc'] = jogos_realizados['DateUtc'] - pd.Timedelta(hours=3)
jogos_future['DateUtc'] = jogos_future['DateUtc'] - pd.Timedelta(hours=3)
#Criar as tabelas
jogos_realizados.to_sql('jogos_realizados', conn, if_exists='replace', index=True)
jogos_future.to_sql('jogos_future', conn, if_exists='replace', index=True)
elo_df.to_sql('elo_df', conn, if_exists='replace', index=True)
#Fechar a conexão com o banco de dados
conn.close()
print("Dados armazenados com sucesso no banco de dados")