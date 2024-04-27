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
# Neo4j

from neo4j import GraphDatabase

class SpotifyLoader:
    def __init__(self, uri, user, password):
        print(f"Connecting to Neo4j at {uri} with username {user}")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        print("Closing connection")
        self.driver.close()

    def load_spotify_data(self, file_url):
        print(f"Loading data from {file_url}")
        with self.driver.session() as session:
            session.execute_write(self._load_data, file_url)
            print("Data load complete")

    @staticmethod
    def _load_data(tx, file_url):
        print("Executing Cypher query")
        query = """
        LOAD CSV WITH HEADERS FROM $file_url AS row
        MERGE (n1:Artists {name: row.artists})
        MERGE (n2:Album {name: row.album_name})
        MERGE (n3:Track {
            name: row.track_name, 
            genre: row.track_genre, 
            popularity: toInteger(row.popularity), 
            duration: toInteger(row.duration_ms), 
            danceability: toFloat(row.danceability), 
            energy: toFloat(row.energy),
            loudness: toFloat(row.loudness),
            speechiness: toFloat(row.speechiness), 
            acousticness: toFloat(row.acousticness), 
            instrumentalness: toFloat(row.instrumentalness), 
            liveness: toFloat(row.liveness), 
            valence: toFloat(row.valence), 
            tempo: toFloat(row.tempo)
        })
        MERGE (n1)-[:featured]->(n2)
        MERGE (n3)-[:part_of]->(n2)
        """
        tx.run(query, file_url=file_url)

if __name__ == "__main__":
    uri = "neo4j://localhost:7687"
    user = "neo4j"
    password = "12345678" 
    file_url = "http://localhost:11001/project-cc095a88-cf6a-4151-b5fa-48307493a4ea/spotifydata.csv"

    loader = SpotifyLoader(uri, user, password)
    try:
        loader.load_spotify_data(file_url)
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        loader.close()

#%% KNN of Artists

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
import pandas as pd
import string
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
import seaborn as sns

df = pd.read_csv("/spotifydata.csv")
categorical = df['track_genre'].map({char : i for i, char in enumerate(string.ascii_uppercase)})
numerical=[feature for feature in df.columns if feature not in categorical]

label_encoder = LabelEncoder()

X = df[['popularity','popularity','duration_ms','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']]
y = df[['track_name', 'artists','album_name']]

X_train, X_test, y_train, y_test = train_test_split( 
             X, y, test_size = 0.2, random_state=42) 

# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)

distances, indices = knn.kneighbors(X_train)
avg_distances = np.mean(distances, axis=1)
avg_cluster_distance = np.mean(avg_distances)

print(avg_cluster_distance)

cluster_labels = knn.predict(X_train)

unique_labels = np.unique(cluster_labels)

cluster_centroids = np.zeros((len(unique_labels), X_train.shape[1]))

for i, label in enumerate(unique_labels):
    cluster_points = X_train[cluster_labels == label]
    cluster_centroids[i] = np.mean(cluster_points, axis=0)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cluster_distances_df, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Pairwise Distance Heatmap between Clusters")
plt.xlabel("Cluster")
plt.ylabel("Cluster")
plt.show()
