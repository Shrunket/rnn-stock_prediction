import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset= pd.read_csv('Google_Stock_Price_Train.csv')
train = dataset.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range= (0,1))
train_scaled= sc.fit_transform(train)

# Creating dataset with 60 timesteps and 1 output
X_train= []
y_train= []
for i in range(60,1258):
    X_train.append(train_scaled[i-60:i, 0])
    y_train.append(train_scaled[i, 0])
    
X_train, y_train= np.array(X_train), np.array(y_train)

# Adding a dimension using reshape
X_train= np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

# Building RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing RNN
regressor= Sequential()

# Add first LSTM Layer and Dropout regularisation
regressor.add(LSTM(units= 50, return_sequences= True, input_shape= (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

# Add Second LSTM Layer and Dropout regularisation
regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(0.2))

# Add Third LSTM Layer and Dropout regularisation
regressor.add(LSTM(units= 50, return_sequences= True))
regressor.add(Dropout(0.2))

# Add Fourth LSTM Layer and Dropout regularisation
regressor.add(LSTM(units= 50))
regressor.add(Dropout(0.2))

# Add Output Layer
regressor.add(Dense(units= 1))

# Compilling 
regressor.compile(optimizer='adam', loss= 'mean_squared_error')

# Fitting model to train dataset
regressor.fit(X_train, y_train, epochs= 100, batch_size= 32)

# Getting real stock price 2017
dataset_test= pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Predicting stock price
dataset_total= pd.concat((dataset['Open'], dataset_test['Open']), axis= 0)
inputs= dataset_total[len(dataset_total)- len(dataset_test)- 60:].values
inputs= inputs.reshape(-1,1)
inputs= sc.transform(inputs)

X_test= []
for i in range(60,80):
    X_test.append(inputs[i-60:i, 0])
    
X_test= np.array(X_test)
X_test= np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price= regressor.predict(X_test)


predicted_stock_price= sc.inverse_transform(predicted_stock_price)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae= mean_absolute_error(real_stock_price, predicted_stock_price)
rmse= mean_squared_error(real_stock_price, predicted_stock_price, squared= False)
print('MAE: ', mae, 'RMSE: ', rmse)

# Visualizing Prediction with real dataset

plt.plot(real_stock_price, color= 'blue', label= 'Real Stock Price')
plt.plot(predicted_stock_price, color= 'red', label= 'Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


