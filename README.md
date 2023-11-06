# Projeto de Previsão xG

## Descrição do Projeto

Este projeto consiste em um modelo de Machine Learning Random Forest para prever o placar de uma partida. Atualmente, o modelo trabalha com os seguintes campeonatos: NBA, NFL, NHL.

## Estrutura do Projeto

O projeto está dividido em 4 arquivos:

- main.py: Arquivo principal do projeto, onde é feita a chamada das funções e a execução do modelo.
- nba.py: Prevê o placar para a NBA.
- nfl.py: Prevê o placar para a NFL.
- nhl.py: Prevê o placar para a NHL.

## Conteúdo do Projeto

* Coleta dados
* Análise de Dados
* Classificação Elo
* Previsão de Resultados
* Treinamento do Modelo
* Banco de Dados
* Requisitos
* Como Usar
* Autor

## Coleta de Dados
Os dados dos jogos da NFL são coletados de várias temporadas usando URLs de feeds JSON. A coleta é feita de forma assíncrona para melhorar a eficiência. Os dados são do site https://fixturedownload.com/.

## Análise de Dados
Os dados coletados são analisados, incluindo estatísticas descritivas dos jogos realizados.

## Classificação Elo
É implementado um sistema de classificação Elo para calcular as classificações Elo de cada time com base em jogos passados.

## Previsão de Resultados
São feitas previsões dos resultados de jogos futuros com base nas classificações Elo dos times.

## Treinamento do Modelo
São treinados modelos de regressão usando a biblioteca scikit-learn para prever pontuações de times em jogos futuros.

## Banco de Dados
Os dados coletados e as previsões são armazenados em um banco de dados SQLite 

## Requisitos
* Python 3.x
* Bibliotecas Python, incluindo pandas, scikit-learn, requests, concurrent.futures, sqlite3, seaborn, scipy, numpy, e outras

## Como Usar
Para executar o projeto, basta executar o arquivo main.py.

## Autor
João Victor Guimarães Florêncio
