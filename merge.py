import csv

with open("data/prices.csv", 'a', newline='') as f:
    writer = csv.writer(f)
    with open("data/prices_append.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = True
        for row in reader:
            if header:
                header = False
                continue
            writer.writerow(row)