import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data\\lending_club_loan_two.csv')
#me figuring out what to drop and what to encode
'''
print("term unqiue: ",len(pd.unique(df['term'])))
print("grade unqiue: ",len(pd.unique(df['grade'])))
print("emp_length: ",len(pd.unique(df['emp_length'])))
print("home_owndership: ",len(pd.unique(df['home_ownership'])))
print("verification_status: ",len(pd.unique(df['verification_status'])))
print("loan_stauts: ",len(pd.unique(df['loan_status'])))
print("purpose: ",len(pd.unique(df['purpose'])))
print("application_type: ",len(pd.unique(df['application_type'])))
'''
df = df.drop(columns=["sub_grade","emp_title","issue_d","title","earliest_cr_line","address","initial_list_status"])

#uhh method i found online for easy one hot encoding
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 
#
features_to_encode = ['term','grade','emp_length','home_ownership','verification_status',"purpose","application_type"]
for feature in features_to_encode:
    df = encode_and_bind(df,feature)

X = df.drop(columns=["loan_status"])
#we act gonna convert all the ones that are fully paid to 1, and the other to 0
Y = df[['loan_status']]
Y.replace({'loan_status':{'Full Paid':1}},inplace=True)
Y.replace({'loan_status':{'Charged Off':0}},inplace=True)
Y.loc[]
print(Y.describe())
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
scaler  = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose =1, patience=25)
model = Sequential([
    Dense(58,activation='relu'),
    Dropout(0.3),
    Dense(25,activation='relu'),
    Dropout(0.3),
    Dense(10,activation='relu'),
    Dropout(0.3),
    Dense(1,activation='sigmoid')
])
'''