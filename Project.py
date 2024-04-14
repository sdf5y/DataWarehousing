#%% libraries
import numpy as np
import pandas as pd
#!pip install faker
from faker import Faker
#!pip install random
import random
#!pip install datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
#!pip install seaborn
import seaborn as sns
#!pip install scikit-learn
#!pip install plotly

#%% Load Dataset

data_df = pd.read_csv('spotifydata.csv', index_col=0) 

#%% Summary Statistics 

# Calculating summary statistics for each numeric column
summary_stats = data_df.describe(include='all')

# Printing the summary statistics
print(summary_stats)


#%%
#Correlation Analysis

data_df[data_df.isnull().any(axis=1)] #looks like we have 1 NA 
#data_df = data_df.dropna(axis=0) #lets drop this one null

data_df.describe().round(0)
data_df.shape
data_df.dtypes

plt.figure(figsize=(10, 8))
sns.heatmap(data_df.select_dtypes(include=["int", "float"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Analysis')
plt.show()

# %%
data_df.columns
# %%
data_df['duration_ms'] = (data_df['duration_ms'] / 1000)
data_df.rename({'duration_ms': 'duration_sec'}, axis=1, inplace=True)
data_df.head()
# %%
sorted_df = data_df.sort_values('popularity', ascending = False)
sorted_df.head()
# %% Top Ten Artists without Collaborations
df_artists= sorted_df.drop_duplicates(subset='artists',keep='first')

colab_index = df_artists['artists'].str.find(';')
df_colab_artists = [df_artists['artists'].loc[i] for i in colab_index.index]
single_artists_df = df_artists[df_artists['artists'].str.contains(";") == False]

top_10_artists = single_artists_df.nlargest(10, 'popularity')
top_10_artists

# %% Top ten Artists including collaborations

top_10_artists = df_artists.nlargest(10, 'popularity')
top_10_artists.head()

# %% Top ten Artists including collaborations
top_10_artists = df_artists.nlargest(10, 'popularity')
top_10_artists.head()

#%%
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'

fig1 = px.bar(top_10_artists, x='popularity', y='artists', text='popularity',
              color='popularity',color_continuous_scale='Viridis')  
fig1.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=2,
                    opacity=0.8,)
fig1.update_layout(title_text='Top 10 Artists in terms of Popularity', yaxis_title='Artists Names',
                   xaxis_title='Popularity (from 0-100)')
fig1.show()

#%% 
colab_artist = []
artist_map = []
for i in range(len(data_df)):
     cell = data_df.iloc[i]['artists']
     cell1 = data_df.iloc[i]['artists']
     if isinstance(cell, str) and ";" in cell:
          cell1 = np.ma.masked_where(isinstance(cell, str) and ";" in cell, cell)
          cell = cell.split(";")
     colab_artist.append(cell)
     artist_map.append(cell1)

#%% Unique artists and popularity

total_weight = sum(df_artists['popularity'])
normalized_weights = [weight / total_weight for weight in df_artists['popularity']]
artists_list = df_artists['artists'].tolist()

#%% Create fake user histories
from faker import Faker
import random

Faker.seed(65678902)
fake = Faker()
data = []

for i in range(1, 30):
    username = fake.first_name()
    email = fake.email()
    country = fake.country()
    artist_plays_list = []
    for ii in range(1, 10):
        artist_plays = random.choices(artists_list, weights=normalized_weights, k=1)
        artist_plays_list.append(artist_plays)
    data.append((username, email, country, artist_plays_list))

# %%
# Selecting Numerical columns for further analysis
num_cols = data_df[data_df.columns[(data_df.dtypes == 'float64') | (data_df.dtypes == 'int64')]]
num_cols.shape
# %%
num_cols.info()
# %%
#Checking distribution of numerical columns
sns.set_style('darkgrid')
sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
num_cols.hist(figsize=(20,15), bins=30, xlabelsize=8, ylabelsize=8)
plt.tight_layout()
plt.show()
# %%
#Logistical Regression for Mode
#!pip install statsmodel.api
import statsmodels.api as sm
from statsmodels.formula.api import glm

model = glm(formula='mode ~ popularity + energy', 
            data = num_cols, family = sm.families.Binomial())
model_fit = model.fit()
print(model_fit.summary())

# %%
num_cols.head

# %% Linear Regression to Predict Popularity by sonic attributes only
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df_artists.drop(columns=['popularity', 'track_id', 'artists', 'album_name', 'track_name', 
                             'mode', 'track_genre', 'explicit', 'key', 'valence', 'time_signature'])
y = df_artists['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

X_train = X_train.astype(int)
X_test = X_test.astype(int)

model = sm.OLS(y_train, X_train).fit()
print(model.summary())

# %% Linear Regression to Predict Popularity by Cultural attributes only
import statsmodels.api as sm

X = df_artists[['explicit', 'danceability', 'energy', 'key', 
       'mode', 'time_signature', 'track_genre']]
y = df_artists['popularity']

categorical_columns = ['track_genre', 'explicit', 'key'] 
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

X_train = X_train.astype(int)
X_test = X_test.astype(int)

model = sm.OLS(y_train, X_train).fit()
print(model.summary())
# %%
