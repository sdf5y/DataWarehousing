# %%

from pymongo import MongoClient
from faker import Faker
import random
import pandas as pd
import numpy as np

# Functions to Generate data
fake = Faker()

def generate_dataset(num_rows):
    rows = []
    for _ in range(num_rows):
        from_city = fake.city()
        to_city = fake.city()
        while to_city == from_city:
            to_city = fake.city()
        distance = random.randint(50, 1000)
        row = {"FromCity": from_city, "ToCity": to_city, "Distance": distance}
        rows.append(row)
    return rows

# Connecting to MongoDB
CONNECTION_STRING = "mongodb://localhost:27017/"
client = MongoClient(CONNECTION_STRING)
db = client['geography']
collection = db['city_distances']

# Generating data and inserting into MongoDB
dataset = generate_dataset(100000)
result = collection.insert_many(dataset)
print(f"Inserted {len(result.inserted_ids)} documents.")

for doc in collection.find().limit(5):
    print(doc)

client.close()

#%% Adding the relevant cities as per the instructions.
data_atlanta = {
    'FromCity': ['Benderchester', 'Atlanta', 'Atlanta', 'Joshuaport', 'Atlanta','Lake Jorgeberg', 'Atlanta', 'Atlanta', 'Port Heidi', 'Atlanta'],
    'ToCity': ['Atlanta', 'Montogomery', 'Pricefort', 'Atlanta', 'Hinesberg','Atlanta','North Arianastad','Phillipsfurt','Atlanta','West Ericton'],
    'Distance': [100, 150, 200, 250, 300,350,400,450,500,550]
}

data_frankfurt =  {
    'FromCity': ['Port Mary', 'Frankfurt','East Tiffany','Frankfurt','Frankfurt','Bethanyview','Frankfurt','Frankfurt','South Amy',
                              'Frankfurt'],
    'ToCity': ['Frankfurt', 'Turnerfurt', 'Frankfurt', 'North Lisa', 'Adamsshire','Frankfurt','Robertfort','New Kimberg','Frankfurt','Maryberg'],
    'Distance': [100, 150, 200, 250, 300,350,400,450,500,550]
}

data_amman =   {'FromCity': ['Amman', 'Amman', 'Amman', 'Amman', 'Amman','Amman', 'Amman', 'Amman', 'Amman', 'Amman'],
    'ToCity': ['North Sara', 'Peterton', 'Kathyton', 'Coxborough', 'Donaldport','North Paulside','Garzashire','South Ashley','Melissaview','Lake Jacob'],
    'Distance': [103, 156, 273, 255, 389,353,410,454,523,557]
               }

df_atlanta = pd.DataFrame(data_atlanta)
df_frankfurt = pd.DataFrame(data_frankfurt)
df_amman = pd.DataFrame(data_amman)

print(df_amman,df_atlanta,df_frankfurt)

#%%
#Generating the unique cities and distances

df = pd.DataFrame()
df = pd.concat([df, df_atlanta, df_frankfurt, df_amman], ignore_index=True)
print(df)

#%% Marging and Saving to csv
main_df = pd.DataFrame(dataset, columns=["FromCity", "ToCity", "Distance"])
print(main_df)
#%%
dataset = pd.concat([main_df, df], ignore_index=True)
print(dataset)

#final_df.to_csv('city_distances.csv', index=False)

# %%
#Task
# 1 List Roads from/to 'Atlanta' with Distances and Destinations

import time
from pymongo import MongoClient

# Connecting to MongoDB
client = MongoClient("mongodb://localhost:27017/")
collection = client['geography']['city_distances']

start_time = time.time()

atlanta_roads = list(collection.find({"$or": [{"FromCity": "Atlanta"}, {"ToCity": "Atlanta"}]}))

elapsed_time = time.time() - start_time  
print(f"Time taken to execute the query: {elapsed_time} seconds")

# Printing results
for road in atlanta_roads:
    print(f"From: {road['FromCity']} - To: {road['ToCity']} - Distance: {road['Distance']} km")

client.close()

#Time taken to execute the query: 1.7097489833831787 seconds

# %%
#2 Find Roads Longer than 150 km, with Details

import time
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
collection = client['geography']['city_distances']

start_time = time.time()
# Querying roads longer than 150 km
long_roads = list(collection.find({"Distance": {"$gt": 150}}))

elapsed_time = time.time() - start_time 
print(f"Time taken to execute the query: {elapsed_time} seconds")

# Printing results
for road in long_roads:
    print(f"From: {road['FromCity']} - To: {road['ToCity']} - Distance: {road['Distance']} km")

client.close()

#Time taken to execute the query: 4.051257848739624 seconds

# %%
#3 Total Road Length Connected to 'Frankfurt'

import time
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
collection = client['geography']['city_distances']

start_time = time.time()
# Aggregating total road length connected to 'Frankfurt'
total_length = collection.aggregate([
    {"$match": {"$or": [{"FromCity": "Frankfurt"}, {"ToCity": "Frankfurt"}]}},
    {"$group": {"_id": None, "TotalDistance": {"$sum": "$Distance"}}}
])

elapsed_time = time.time() - start_time  
print(f"Time taken to execute the query: {elapsed_time} seconds")
# Printing total road length
for result in total_length:
    print(f"Total road length connected to Frankfurt: {result['TotalDistance']} km")

client.close()

#Time taken to execute the query: 5.0743818283081055 seconds

# %%
#4 Determine Shortest and Longest Road from 'Amman'

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
collection = client['geography']['city_distances']

start_time = time.time()
# Finding the shortest road from 'Amman'
shortest_road = collection.find({"FromCity": "Amman"}).sort("Distance", 1).limit(1)
longest_road = collection.find({"FromCity": "Amman"}).sort("Distance", -1).limit(1)

elapsed_time = time.time() - start_time  
print(f"Time taken to execute the query for shortest road: {elapsed_time} seconds")
print(f"Time taken to execute the query for longest road: {elapsed_time} seconds")

# Printing the results
print("Shortest Road from 'Amman':")
for road in shortest_road:
    print(f"From: {road['FromCity']} - To: {road['ToCity']} - Distance: {road['Distance']} km")

print("Longest Road from 'Amman':")
for road in longest_road:
    print(f"From: {road['FromCity']} - To: {road['ToCity']} - Distance: {road['Distance']} km")

client.close()

#Time taken to execute the query for shortest road: 0.007380008697509766 seconds
#Time taken to execute the query for longest road: 0.007380008697509766 seconds

# %%

import time 
from py2neo import Graph

# Connect to Neo4j database
graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

# Load CSV data
load_csv_query = """
LOAD CSV WITH HEADERS FROM 'http://localhost:11001/project-5c8381fb-01a5-44cd-9296-2c042124f5b3/city_distances.csv' AS row
WITH row
MERGE (c1:City {name: row.FromCity})
MERGE (c2:City {name: row.ToCity})
MERGE (c1)-[r:ROAD {Distance: row.Distance}]->(c2)
"""
start_time = time.time()
graph.run(load_csv_query)
print(f"CSV Load Time: {time.time() - start_time:.2f} seconds")

#Time taken to load CSV: 1943.05 seconds

#%%
# 1 List Roads from/to 'Atlanta' with Distances and Destinations

query1 = """
MATCH (atlanta:City {name: 'Atlanta'})-[road:ROAD]->(destination)
RETURN atlanta.name AS fromCity, destination.name AS toCity, road.Distance AS distance
"""
start_time = time.time()
result1 = graph.run(query1)
print("Roads from/to Atlanta:")
for record in result1:
    print(record)
print(f"Query Time: {time.time() - start_time} seconds")

#Time taken to execute the query: 0.038472890853881836 seconds

#%%
#2 Find Roads Longer than 150 km, with Details

query2 = """
MATCH (city:City)-[road:ROAD]->(destination)
WHERE toInteger(road.Distance) > 150
RETURN city.name AS fromCity, destination.name AS toCity, road.Distance AS distance
"""
start_time = time.time()
result3 = graph.run(query2)
print("\nRoads longer than 150 km:")
for record in result3:
    print(record)
print(f"Query Time: {time.time() - start_time} seconds")

#Time taken to execute the query: 3.2383840084075928 seconds

#%%
#3 Total Road Length Connected to 'Frankfurt'

query3 = """
MATCH (city:City {name: 'Frankfurt'})<-[road:ROAD]->(otherCity)
RETURN SUM(toInteger(road.Distance)) AS totalRoadLength
"""
start_time = time.time()
result2 = graph.run(query3)
print("\nTotal road length connected to Frankfurt:")
print(result2)
print(f"Query Time: {time.time() - start_time} seconds")

#Time taken to execute the query: 0.04621005058288574 seconds

#%%
#4 Determine Shortest Road from 'Amman'

query4_shortest = """
MATCH (city:City{name:'Amman'})-[road:ROAD]->(destination)
WITH destination, toInteger(road.Distance) AS distance
ORDER BY distance
LIMIT 1
RETURN 'Amman' AS fromCity, destination.name AS toCity, distance AS shortestDistance
"""
start_time = time.time()
result4_shortest = graph.run(query4_shortest)
print("\nShortest road from Amman:")
print(result4_shortest)
print(f"Query Time: {time.time() - start_time} seconds")

#Time taken to execute the query: 0.04681897163391113 seconds

#%%
# Determine Longest Road from 'Amman'

query4_longest = """
MATCH (city:City{name:'Amman'})-[road:ROAD]->(destination)
WITH destination, toInteger(road.Distance) AS distance
ORDER BY distance DESC
LIMIT 1
RETURN 'Amman' AS fromCity, destination.name AS toCity, distance AS longestDistance
"""
start_time = time.time()
result4_longest = graph.run(query4_longest)
print("\nLongest road from Amman:")
print(result4_longest)
print(f"Query Time: {time.time() - start_time} seconds")

#Time taken to execute the query: 0.06295394897460938 seconds

# %%
# Bar graph
import matplotlib.pyplot as plt

queries = ['List Roads', 'Find Roads > 150km', 'Total Road Length', 'Shortest Road', 'Longest Road']
mongo_times = [1.70, 4.05, 5.07, 0.01, 0.01]  
neo4j_times = [0.03, 3.23, 0.04, 0.04, 0.06]  

x = range(len(queries)) 
width = 0.35  

fig, ax = plt.subplots()
rects1 = ax.bar([xi - width/2 for xi in x], mongo_times, width, label='MongoDB')
rects2 = ax.bar([xi + width/2 for xi in x], neo4j_times, width, label='Neo4j')

ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Query Execution Time Comparison between MongoDB and Neo4j')
ax.set_xticks(x)
ax.set_xticklabels(queries, rotation=45, ha="right")
ax.legend()

# To display the labels on the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 6)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()

# %%
#Part 2
#Tasks

from neo4j import GraphDatabase

# Define a class to handle the Neo4j connection and queries
class Neo4jService:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query):
        with self.driver.session() as session:
            start_time = time.time()
            result = session.run(query)
            records = [record for record in result]
            duration = time.time() - start_time
            return records, duration
             
#%%
# Connection details (replace these with your actual connection details)
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

# Instantiate the service
neo4j_service = Neo4jService(uri, user, password)


#%%
# DFS on Atlanta
query_dfs_atlanta = """
MATCH (source:City {name: 'Atlanta'})
CALL gds.dfs.stream('City', {sourceNode: source})
YIELD path
RETURN [node in nodes(path) | node.name] AS traversalSequence
"""
print(neo4j_service.execute_query(query_dfs_atlanta))
records, duration = neo4j_service.execute_query(query_dfs_atlanta)
print(f"DFS on Atlanta executed in {duration} seconds")

#DFS on Atlanta executed in 0.26135802268981934 seconds

#%%
# DFS on Frankfurt
query_dfs_Frankfurt = """
MATCH (source:City {name: 'Frankfurt'})
CALL gds.dfs.stream('City', {sourceNode: source})
YIELD path
RETURN [node in nodes(path) | node.name] AS traversalSequence
"""
print(neo4j_service.execute_query(query_dfs_Frankfurt))
records, duration = neo4j_service.execute_query(query_dfs_Frankfurt)
print(f"DFS on Frankfurt executed in {duration} seconds")

#DFS on Frankfurt executed in 0.2294158935546875 seconds

#%%
# BFS on Atlanta
query_bfs_atlanta = """
MATCH (source:City {name: 'Atlanta'})
CALL gds.bfs.stream('City', {sourceNode: source})
YIELD path
RETURN [node in nodes(path) | node.name] AS traversalSequence
"""
print(neo4j_service.execute_query(query_bfs_atlanta))
records, duration = neo4j_service.execute_query(query_bfs_atlanta)
print(f"BFS on Atlanta executed in {duration} seconds")

#BFS on Atlanta executed in 0.22919511795043945 seconds

#%%
# BFS on Frankfurt
query_bfs_Frankfurt = """
MATCH (source:City {name: 'Frankfurt'})
CALL gds.bfs.stream('City', {sourceNode: source})
YIELD path
RETURN [node in nodes(path) | node.name] AS traversalSequence
"""
print(neo4j_service.execute_query(query_bfs_Frankfurt))
records, duration = neo4j_service.execute_query(query_bfs_Frankfurt)
print(f"BFS on Frankfurt executed in {duration} seconds")

#BFS on Frankfurt executed in 0.5094900131225586 seconds

# Clean up
neo4j_service.close()

# %%
# Comparing DFS and BFS

import matplotlib.pyplot as plt

times_atlanta = {'DFS': 0.26, 'BFS': 0.22}
times_Frankfurt = {'DFS': 0.22, 'BFS': 0.50 }

# Labels for each bar group
labels = ['DFS', 'BFS']

atlanta_times = [times_atlanta['DFS'], times_atlanta['BFS']]
Frankfurt_times = [times_Frankfurt['DFS'], times_Frankfurt['BFS']]

indices = range(len(labels))

fig, ax = plt.subplots()
bar_width = 0.35 

# Plotting both Atlanta and Frankfurt bars
atlanta_bars = ax.bar(indices, atlanta_times, bar_width, label='Atlanta')
Frankfurt_bars = ax.bar([p + bar_width for p in indices], Frankfurt_times, bar_width, label='Frankfurt')

ax.set_xlabel('Algorithm')
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('DFS vs BFS Execution Time Comparison')
ax.set_xticks([p + bar_width / 2 for p in indices])
ax.set_xticklabels(labels)
ax.legend()

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(atlanta_bars)
add_labels(Frankfurt_bars)

plt.show()

# %%
