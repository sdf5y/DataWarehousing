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

final_df = pd.concat([main_df, df], ignore_index=True)

#final_df.to_csv('city_distances.csv', index=False)
#%% Find Roads with Atlanta in it
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['citydistance']
roads_collection = db['roads']

atlanta_roads = roads_collection.find({'$or': [{'source': 'Atlanta'}, {'destination': 'Atlanta'}]})

for road in atlanta_roads:
    print(f"From {road['source']} to {road['destination']} with distance {road['distance']} km")


#%% Find Roads longer than 150 km
long_roads = roads_collection.find({'distance': {'$gt': 150}})

for road in long_roads:
    print(f"Road from {road['source']} to {road['destination']} with distance {road['distance']} km")

#%% Find the total road lengths to Frankfurt
from bson.son import SON

pipeline = [
    {'$match': {'$or': [{'source': 'Frankfurt'}, {'destination': 'Frankfurt'}]}},
    {'$group': {'_id': None, 'total_length': {'$sum': '$distance'}}}
]

result = list(roads_collection.aggregate(pipeline))
total_length = result[0]['total_length'] if result else 0


#%% Longest and Shortest Roads to Amman

shortest_road = roads_collection.find_one({'$or': [{'source': 'Amman'}, {'destination': 'Amman'}]}, sort=[('distance', 1)])

longest_road = roads_collection.find_one({'$or': [{'source': 'Amman'}, {'destination': 'Amman'}]}, sort=[('distance', -1)])

