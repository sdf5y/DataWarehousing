# %%

from pymongo import MongoClient
from faker import Faker
import random

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

#Time taken to execute the query: 0.1284027099609375 seconds

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

#Time taken to execute the query: 0.6547698974609375 seconds

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

#Time taken to execute the query: 0.12424707412719727 seconds

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
print(f"Time taken to execute the query: {elapsed_time} seconds")

# Printing the results
print("Shortest Road from 'Amman':")
for road in shortest_road:
    print(f"From: {road['FromCity']} - To: {road['ToCity']} - Distance: {road['Distance']} km")

print("Longest Road from 'Amman':")
for road in longest_road:
    print(f"From: {road['FromCity']} - To: {road['ToCity']} - Distance: {road['Distance']} km")

client.close()

#Time taken to execute the query: 0.0001270771026611328 seconds

# %%
