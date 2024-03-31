#%% libraries
import numpy as np
import pandas as pd

!pip install faker
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
# %%

df_artists=sorted_df.drop_duplicates(subset='artists',keep='first')
df_artists.head()
# %%
top_10_artists = df_artists.nlargest(10, 'popularity')
top_10_artists.head()
# %%

# %%


