import os
import csv

files = os.listdir("runs/train/")

for name in files:

    subfiles = os.listdir("runs/train/" + name + "/")

    for subfile in subfiles:

        rows = []
        #print(subfile)
        with open("runs/train/" + name + "/" + subfile + "/results.csv", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)

        average = 0.0
        averageSample = 5
        for i in range(averageSample):
            average += float(rows[-averageSample][6])

        average /= averageSample

        print(subfile + ": " + str(average))