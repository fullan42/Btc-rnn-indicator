# Btc-rnn-indicator
﻿#	 Project outline:
﻿#	Problem definition: Complexity of stock prices for traders and trade platforms.
﻿#	Purpose: Develop price indicator with RNN try to define next day price 
﻿#	Neural network method that will be use: Recurrent neural networks method.
﻿#	Reason than choosing RNN: particular have received the most success when working with sequences of words and paragraphs, generally called natural language processing. So, input and output style of this method fitting perfectly to stock and bitcoin datasets. Especially many to one perspective.
﻿# 2. Description of your dataset and how it was obtained.
﻿# 2.0 Material and Method
The purpose of this paper is to develop a model that takes a sequence of time series data as input, processes it, trains the recurrent neural network, and predicts the future price for the Bitcoin cryptocurrency. The complete process is divided into. four major sections: getting past-time data, preparing the data for training and testing, creating and training an (LSTM) Recurrent Neural Network model for predicting the price of a cryptocurrency, and predicting the prices using the trained model. We utilized the deep learning framework TensorFlow (Kanagachidambaresan, 2021) to address the bitcoin price prediction. A dataset from (Investing.com) was used for training and testing. The created approach can forecast prices based on the last 30 days.
﻿# 2.1. Data Preparation
Data in the LSTM model requires to be in the form of sequences of XS and YS. Where in this model X represents the last 30 days' prices as an input and y represents the 31st-day prices as an output. Because the LSTM algorithm is based on neural networks, standardizing or normalizing the data is required for a better performance and optimization that will upgradable in the future.
﻿#	2.1.1. Data Normalization
 Our model is built on an RNN with LSTM layers that transforms input vectors into vectors with entries in the [0,1] range using the sigmoid activation function. Because the denominator greater than nominator at all times, we apply normalization to optimize model features. the output will be 0-1 . we will be using the MinMaxScaler class from the sklearn.preprocessing library in our code . We did research about normalization in pyhton and we find this library we think that it is useful so we decide to use it.
 tests
![image](https://github.com/fullan42/Btc-rnn-indicator/assets/53313497/0f04e316-c362-41d7-b5f5-7aa721bb4908)

![image](https://github.com/fullan42/Btc-rnn-indicator/assets/53313497/75305646-c2b6-476d-abd5-220690dcfb2c)

![image](https://github.com/fullan42/Btc-rnn-indicator/assets/53313497/0fe1f08c-91c1-4318-a004-fc15126efbc9)
