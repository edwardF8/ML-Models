import numpy
from matplotlib import pyplot
import csv
## work with data, https://www.kaggle.com/datasets/mirichoi0218/insurance
data = []
with open('NoLib\medical_cost.csv','r') as file: #opens csv and saves under file
    csv_reader = csv.reader(file, delimiter=',') #processes csv and takes out comma
    for row in csv_reader:
        data.append(row)
for row in data:
    row[3] /= 53.1 #scaling BMI to values in between 0 and 1
    row[7] /= 63770.42801 #scaling price, will multiply label by 63770.42801
    row[4] /= 5.0
    if(row[2] == "female"):
        row[2] = 1
    else:
        row[2] = 0
    if(row[5] == 'yes'):
        row[5] = 1
    else:
        row[5] = 0
normalset = data[:1001]
testset = data[1001:]


### 1339 inputs in this set
#spliting data up into different sets
for row in normalset:
    print(row)
    
    
print("__________________________")
for row in testset:
    print(row)
    
## define model