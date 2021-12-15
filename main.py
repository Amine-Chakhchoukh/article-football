import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from urllib.request import urlopen

import seaborn as sns
import matplotlib.pyplot as plt

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)
pd.set_option('display.max_rows', None)

url = 'https://en.wikipedia.org/wiki/List_of_UEFA_European_Championship_penalty_shoot-outs#Complete_list'
html = urlopen(url)
soup = BeautifulSoup(html, 'html.parser')

tables = soup.find_all('table')


# Create arrays to hold the data we extract
Nos = []
rates = []
Editions = []
Winners = []
Losers = []
Win_scorers = []
Win_losers = []
Los_scorers = []
Los_losers = []
Rounds = []
n_kicks = []

table = tables[0]

rows = table.find_all('tr')

rowspan = 1
rowspan_1 = 1

for row in rows:
    cells = row.find_all('td')
    if len(cells) > 1:

        # Column with numbers
        No = cells[0]
        Nos.append(No.text.strip())

        # Column with Edition
        if rowspan == 1:
            Edition = cells[1]
            Editions.append(Edition.text.strip())
            if Edition.has_attr("rowspan"):
                rowspan = int(Edition["rowspan"])
        else:
            cells.insert(1, Editions[-1])
            Editions.append(Edition.text.strip())
            rowspan = rowspan - 1

        # Column with Winners
        winner = cells[2]
        Winners.append(winner.text.strip())

        # Column with Losers
        loser = cells[4]
        Losers.append(loser.text.strip())

        # Before we proceed, we need to make sure players' names are standardised!
        names_dict = {
            'A. Cole': 'Ashley Cole',
            'Owen': 'Michael Owen',
            'G. Baresi': 'Giuseppe Baresi',
            'J. Olsen': 'Jesper Olsen',
            'R. de Boer': 'Ronald de Boer',
            'F. de Boer': 'Frank de Boer',
            'Pepe': 'Pepe F.',
            'Ricardo': 'Ricardo P.',
            'Xhaka': 'Granit Xhaka'
        }

        # Columns with scorers/losers of the winning team
        Takers = cells[9].find_all('span')
        Win_scorers_thisgame = []
        Win_losers_thisgame = []
        for taker in Takers:
            if not taker.find('span'):
                if taker.find('img')['alt'] == "Penalty scored":
                    if taker.text.strip() in names_dict:
                        Win_scorers_thisgame.append(taker.text.strip().replace(taker.text.strip(), names_dict[taker.text.strip()]))
                    else:
                        Win_scorers_thisgame.append(taker.text.strip())
                else:
                    if taker.text.strip() in names_dict:
                        taker = taker.text.strip().replace(taker.text.strip(), names_dict[taker.text.strip()])
                    Win_losers_thisgame.append(taker.text.strip())
        if Win_scorers_thisgame:
            Win_scorers.append(Win_scorers_thisgame)
        else:
            Win_scorers.append([])
        if Win_losers_thisgame:
            Win_losers.append(Win_losers_thisgame)
        else:
            Win_losers.append([])

        # Columns with scorers/losers of the Losing team
        Takers = cells[10].find_all('span')
        Los_scorers_thisgame = []
        Los_losers_thisgame = []
        for taker in Takers:
            if not taker.find('span'):
                if taker.find('img')['alt'] == "Penalty scored":
                    if taker.text.strip() in names_dict:
                        Los_scorers_thisgame.append(taker.text.strip().replace(taker.text.strip(), names_dict[taker.text.strip()]))
                    else:
                        Los_scorers_thisgame.append(taker.text.strip())
                else:
                    if taker.text.strip() in names_dict:
                        Los_losers_thisgame.append(taker.text.strip().replace(taker.text.strip(), names_dict[taker.text.strip()]))
                    else:
                        Los_losers_thisgame.append(taker.text.strip())
        if Los_scorers_thisgame:
            Los_scorers.append(Los_scorers_thisgame)
        else:
            Los_scorers.append([])
        if Los_losers_thisgame:
            Los_losers.append(Los_losers_thisgame)
        else:
            Los_losers.append([])

        # Column with Round
        if rowspan_1 == 1:
            Round = cells[12]
            Rounds.append(Round.text.strip())
            if Round.has_attr("rowspan"):
                rowspan_1 = int(Round["rowspan"])
        else:
            cells.insert(12, Rounds[-1])
            Rounds.append(Round.text.strip())
            rowspan_1 = rowspan_1 - 1

        # Column with n_kicks
        n_kicks.append(len(Win_scorers_thisgame)+len(Win_losers_thisgame)+len(Los_scorers_thisgame)+len(Los_losers_thisgame))


df = pd.DataFrame(Editions, index=Nos, columns=['Edition'])
df['Winners'] = Winners
df['Losers'] = Losers
df['Winning team: scorers'] = Win_scorers
df['Winning team: losers'] = Win_losers
df['Losing team: scorers'] = Los_scorers
df['Losing team: losers'] = Los_losers
df['Round'] = Rounds
df['game'] = df['Winners'] + ' vs ' + df['Losers'] + ' -- ' + df['Edition'].str[:4]
df['n_kicks'] = n_kicks
df = df.iloc[:-4, :]
print(df)

sns.set_theme()

f, ax = plt.subplots()
sns.barplot(x="n_kicks", y="game", data=df, ci=None)
x = 2 * np.arange(10)
ax.set_xticks(x)
plt.tight_layout()
# plt.show()

# Table for players who scored
df_scorers = df[['Edition']]
df_scorers['scorers'] = df['Winning team: scorers'] + df['Losing team: scorers']
df_scorers['Edition'] = df_scorers['Edition'].str[:4].astype(int)
df_scorers.set_index(['Edition'], inplace=True)
df_scorers = df_scorers.apply(lambda x: x.explode())
df_scorers.reset_index(inplace=True)
df_scorers = df_scorers[df_scorers['Edition'].astype(int) < 2020]
print(df_scorers)
x = df_scorers.scorers.unique()

# Table for players who missed
df_losers = df[['Edition']]
df_losers['losers'] = df['Winning team: losers'] + df['Losing team: losers']
df_losers['Edition'] = df_losers['Edition'].str[:4].astype(int)
df_losers.set_index(['Edition'], inplace=True)
df_losers = df_losers.apply(lambda x: x.explode())
df_losers.reset_index(inplace=True)
df_losers = df_losers[df_losers['Edition'].astype(int) < 2020]

# Players and their age
df_age = pd.read_csv('players.csv')[['PlayerName(Captain)', 'DateofBirth(age)', 'Year']]
df_age['age'] = df_age['DateofBirth(age)'].str[-3:-1].astype(int)
df_age.drop(['DateofBirth(age)'], axis=1, inplace=True)
df_age['player'] = df_age['PlayerName(Captain)'].str.replace('captain', '')
df_age['player'] = df_age['player'].str.replace('(', '')
df_age['player'] = df_age['player'].str.replace(')', '')
df_age['player'] = df_age['player'].astype(str).str.split().str[1]
df_age['PlayerName(Captain)'] = df_age['PlayerName(Captain)'].replace('Pepe', 'Pepe F.').replace('Pepe Reina F.', 'Pepe Reina')
df_age['PlayerName(Captain)'] = df_age['PlayerName(Captain)'].replace('Ricardo', 'Ricardo P.').replace('Ricardo P. Carvalho', 'Ricardo Carvalho').replace('Ricardo P. Cabanas', 'Ricardo Cabanas')


# Players in the two tables may have different names
df_scorers['join'] = 1
df_age['join'] = 1

df_full = df_scorers.merge(df_age, on='join').drop('join', axis=1)
df_age.drop('join', axis=1, inplace=True)
df_full = df_full[df_full['Edition'] == df_full['Year']].drop(columns=['Year'])
df_full['player'] = df_full['player'].astype('str')
df_full['match'] = df_full.apply(lambda x: x['PlayerName(Captain)'].lower().find(x.scorers.lower()), axis=1).ge(0)
df_full = df_full[df_full['match'] == True]
df_full['scored ?'] = 'scored'
df_scored = df_full.drop(columns=['scorers', 'player', 'match']).reset_index(drop=True)
y = df_full.scorers.unique()
print(df_scored)

import numpy as np
main_list = np.setdiff1d(y, x)
print("diff", main_list)

print("x", len(x))
print("y", len(y))

# Players in the two tables may have different names
df_losers['join'] = 1
df_age['join'] = 1

print(df_losers)
df_full = df_losers.merge(df_age, on='join').drop('join', axis=1)
df_age.drop('join', axis=1, inplace=True)
df_full = df_full[df_full['Edition'] == df_full['Year']].drop(columns=['Year'])
df_full['player'] = df_full['player'].astype('str')
df_full['match'] = df_full.apply(lambda x: x['PlayerName(Captain)'].lower().find(x.losers.lower()), axis=1).ge(0)
df_full = df_full[df_full['match'] == True]
df_full['scored ?'] = 'missed'
df_missed = df_full.drop(columns=['losers', 'player', 'match']).reset_index(drop=True)
y = df_full.losers.unique()
print(df_missed)

#  Finally!
df_scored_missed = pd.concat([df_scored, df_missed])
print(df_scored_missed.sort_values("age"))

sns.set_theme()

f, ax = plt.subplots()
ax = sns.countplot(data=df_scored_missed, x="scored ?")
for p in ax.patches:
    ax.annotate("{} ({}%)".format(p.get_height(), format((p.get_height()/194)*100, '.0f')), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                xytext=(0, 10), textcoords='offset points')
# x = 2 * np.arange(10)
# ax.set_xticks(x)
ax.set(xlabel=None)
ax.set(ylim=(0, 170))
plt.tight_layout()
# plt.show()

df_pyramid = df_scored_missed.drop(columns=['Edition', 'PlayerName(Captain)'])
df_pyramid['missed'] = 0
df_pyramid.loc[df_pyramid['scored ?'] == 'missed', 'missed'] = -1
df_pyramid['scored'] = 0
df_pyramid.loc[df_pyramid['scored ?'] == 'scored', 'scored'] = 1
df_pyramid = df_pyramid.groupby("age").sum().reset_index()
df_pyramid['age'] = df_pyramid['age'].apply(str)
print(df_pyramid)

plt.rcParams["figure.figsize"] = (10, 8)

# #Class
# AgeClass = ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
# print(AgeClass)
# AgeClass.reverse()
# print(AgeClass)

# #Chart
# bar_plot = sns.barplot(x='missed', y='age', data=df_pyramid, order=AgeClass, lw=0, palette="PRGn")
# bar_plot = sns.barplot(x='scored', y='age', data=df_pyramid, order=AgeClass, lw=0, palette="Spectral")
# plt.title("Pyramid Age Distribution of Genders", fontsize=22)
# plt.xlabel("Female/Male")
# # plt.show()

# # Prepare Data
# plt.rcParams["figure.figsize"] = (10, 8)
# df = pd.DataFrame({'Age': ['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80-84','85-89','90-94','95-99','100+'],
#                     'Male': [-50000, -61000, -66000, -58000, -44000, -46000, -31000, -23000, -22000, -20000, -17000, -15000, -13000, -11000, -7600, -4600, -1900, -2000, -1160, -1400, -1000],
#                     'Female': [52367, 65000, 67000, 57000, 45000, 47000, 33000, 26000, 25000, 23000, 20000, 16000, 14000, 10000, 9000, 8600, 3900, 2000, 1200, 1800, 0]})
#
# print(df)
# #Class
# AgeClass = ['100+','95-99','90-94','85-89','80-84','75-79','70-74','65-69','60-64','55-59','50-54','45-49','40-44','35-39','30-34','25-29','20-24','15-19','10-14','5-9','0-4']
# #Chart
# bar_plot = sns.barplot(x='Male', y='Age', data=df, order=AgeClass, lw=0, palette="PRGn")
# bar_plot = sns.barplot(x='Female', y='Age', data=df, order=AgeClass, lw=0, palette="Spectral")
# plt.title("Pyramid Age Distribution of Genders", fontsize=22)
# plt.xlabel("Female/Male")
# # plt.show()

# plt.subplot(1, 2, 1) # row 1, col 2 index 1
# g = sns.catplot(x="scored ?", y="age", kind="swarm", data=df_scored_missed).set(xlabel=None)
# y = np.arange(18, 36)
# g.set(yticks=y)
# # plt.tight_layout()
#
# plt.subplot(1, 2, 2) # index 2
# sns.barplot(x="n_kicks", y="game", data=df, ci=None)
# # x = 2 * np.arange(10)
# # ax.set_xticks(x)
# plt.tight_layout()
#
# plt.show()

g = sns.catplot(x="scored ?", y="age", kind="box", data=df_scored_missed).set(xlabel=None)
g.set(yticks=np.arange(18, 36))
plt.show()

y = pd.get_dummies(df_scored_missed['scored ?'])
df_scored_missed = df_scored_missed.join(y)

print(df_scored_missed.head())
print(df_scored_missed.corr(method='pearson'))

