import csv

with open("data/prices.csv", 'a', newline='') as f:
    writer = csv.writer(f)
    with open("data/prices_add.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            writer.writerow(row)