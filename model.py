import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("multiHomePrice.csv")

med = df.age.median()
df.age = df.age.fillna(med)

x = df.iloc[:, 1:4]
y = df.price

model = LinearRegression()
model.fit(x, y)

# using pickle

pickle_file = open('model.pkl', 'wb')
pickle.dump(model, pickle_file)

