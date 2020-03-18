import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import statistics as s

Patient = input("Enter Patient Name")
inputNum = float(input("enter period of breath function"));
inputNum2= float(input("enter difference between max and min"))
df = pd.read_csv("ventilator.csv")

X = df[['period']].values
y = np.log(df[['oxigen_per_lit']].values)
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.0000000000001, random_state=42)

scaler= MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
model = Sequential()
model.add(Dense(8,activation = 'tanh'))
model.add(Dense(4,activation = 'tanh'))
model.add(Dense(2,activation = 'tanh'))
model.add(Dense(1))
model.compile(optimizer= 'rmsprop', loss = 'mse')
model.fit(x=X_train, y =y_train,epochs=250)

kana = np.e**float(model.predict([inputNum]))
donovan = s.mean(df[['oxigen_per_lit']].values.reshape(1432,))
isWonderful = s.stdev(df[['oxigen_per_lit']].values.reshape(1432,))
andShe = s.mean(df[['peaks_diff']].values.reshape(1432,))
shouldKnowThat = s.stdev(df[['peaks_diff']].values.reshape(1432,))
zScore1= (kana - donovan)/isWonderful
zScore2 = (inputNum-andShe)/shouldKnowThat

if(zScore1<(-1.65) or (zScore2<(-1.65))):
    print("Ventilator necessary for " + Patient)
else:
    print("No ventilation necessary")
