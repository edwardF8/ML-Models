import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ML-Models\\data\\lending_club_loan_two.csv')
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
Y.loc[Y['loan_status'] != 'Fully Paid', 'loan_status'] = 0
Y.loc[Y['loan_status'] == 'Fully Paid', 'loan_status'] = 1
Y['loan_status'] = Y['loan_status'].astype(int)
print(X.info())
print(Y.info())
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
scaler  = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose =1, patience=25)
model = Sequential([
    In/
    Dense(58),
    Dropout(0.3),
    Dense(25,activation='sigmoid'),
    Dropout(0.3),
    Dense(10,activation='tanh'),
    Dropout(0.3),
    Dense(1,activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=x_train, y=y_train,epochs=400, batch_size=100, validation_data=(x_test, y_test), callbacks=[early_stop])

loss = pd.DataFrame(model.history.history)
loss.plot()
plt.show()