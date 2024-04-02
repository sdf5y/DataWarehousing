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

#%% Load Dataset

data_df = pd.read_csv('spotifydata.csv', index_col=0) 

#%% Summary Statistics 

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

colab_index = df_artists['artists'].str.find(';')
df_colab_artists = [df_artists['artists'].loc[i] for i in colab_index.index]
single_artists_df = df_artists[df_artists['artists'].str.contains(";") == False]

top_10_artists = single_artists_df.nlargest(10, 'popularity')
top_10_artists

# %% Top ten Artists including collaborations

df_artists= sorted_df.drop_duplicates(subset='artists',keep='first')
df_artists.head()

top_10_artists = df_artists.nlargest(10, 'popularity')
top_10_artists.head()

# %% Top ten Artists including collaborations
top_10_artists = df_artists.nlargest(10, 'popularity')
top_10_artists.head()

# %%
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
# %%

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
