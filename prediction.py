import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv('http://bit.ly/w-data')
print(data.head())

array = data.values
X = array[:, 0]
Y = array[:, -1]

lr = LinearRegression()
lr.fit(X.reshape(-1, 1), Y.reshape(-1, 1))

prediction = lr.predict([[6]])
print('Prediction', prediction)

#saving the model
pickle.dump(lr, open('model.pkl', 'wb'))

#loading the model
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[6]]))