#%%
from faker import Faker
import random
import csv
import numpy as np
import pandas as pd

#%% Functions to Generate data
fake = Faker()

def generate_dataset(num_rows):
    cities = set()
    rows = []
    
    for _ in range(num_rows):
        from_city = fake.city()
        to_city = fake.city()
        while to_city == from_city:
            to_city = fake.city()
        distance = random.randint(50, 1000)  # Random distance between 50 and 1000 km
        row = (from_city, to_city, distance)
        rows.append(row)
        cities.add(from_city)
        cities.add(to_city)
    
    return rows, cities

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
#%% Generating the unique cities and distances

df = pd.DataFrame()
df = pd.concat([df, df_atlanta, df_frankfurt, df_amman], ignore_index=True)

main_dataset, cities = generate_dataset(100000)

#%% Marging and Saving to csv
main_df = pd.DataFrame(main_dataset, columns=["FromCity", "ToCity", "Distance"])

dataset = pd.concat([main_df, df], ignore_index=True)

#final_df.to_csv('city_distances.csv', index=False)
#%% Chaya's Code
# Connecting to MongoDB
CONNECTION_STRING = "mongodb://localhost:27017/"
client = MongoClient(CONNECTION_STRING)
db = client['geography']
collection = db['city_distances']

# Generating data and inserting into MongoDB
#dataset = generate_dataset(100000)
result = collection.insert_many(dataset)
print(f"Inserted {len(result.inserted_ids)} documents.")

for doc in collection.find().limit(5):
    print(doc)

client.close()


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

#Time taken to execute the query: 0.17115116119384766 seconds

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

#Time taken to execute the query: 0.5260400772094727 seconds

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

#Time taken to execute the query for shortest road: 0.0001862049102783203 seconds
#Time taken to execute the query for longest road: 0.0001862049102783203 seconds

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

#Time taken to load CSV: 1549.78 seconds

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
print(f"Query Time: {time.time() - start_time:.2f} seconds")

#Time taken to execute the query: 0.04 seconds

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
print(f"Query Time: {time.time() - start_time:.2f} seconds")

#Time taken to execute the query: 1.54 seconds

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
print(f"Query Time: {time.time() - start_time:.2f} seconds")

#Time taken to execute the query: 0.05 seconds

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
print(f"Query Time: {time.time() - start_time:.2f} seconds")

#Time taken to execute the query: 0.12 seconds

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
print(f"Query Time: {time.time() - start_time:.2f} seconds")

#Time taken to execute the query: 0.20 seconds

# %%
# Bar graph
import matplotlib.pyplot as plt

queries = ['List Roads', 'Find Roads > 150km', 'Total Road Length', 'Shortest Road', 'Longest Road']
mongo_times = [0.17, 4.05, 0.52, 0.01, 0.01]  
neo4j_times = [0.04, 1.54, 0.05, 0.12, 0.20]  

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
