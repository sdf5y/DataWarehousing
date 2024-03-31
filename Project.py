#%% libraries
import numpy as np
import pandas as pd
#!pip install psycopg2 
import psycopg2
#!pip install faker
from faker import Faker
#!pip install random
import random
#!pip install datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
#!pip install seaborn
import seaborn as sns
import os
import math
import scipy.stats
import copy
#!pip install scikit-learn

#%% Load Dataset

data_df = pd.read_csv('spotifydata.csv', index_col=0) 

#%% Summary Statistics 

data_df[data_df.isnull().any(axis=1)] #looks like we have 1 NA 
#data_df = data_df.dropna(axis=0) #lets drop this one null

data_df.describe()

data_df.dtypes
sns.heatmap(data_df.select_dtypes(include=["int", "float"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Analysis')
plt.show()
# %%
