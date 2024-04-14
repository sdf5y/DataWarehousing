#%%
from faker import Faker
import random
import csv

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

def write_to_csv(filename, rows):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["FromCity", "ToCity", "Distance"])
        writer.writerows(rows)
#%% Generating the unique cities and distances, then saving to csv

dataset, cities = generate_dataset(100000)

write_to_csv('city_distances.csv', dataset)

#%%