# Fraud Detection using Self Organizing Map (SOM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dataset/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values # all rows, all columns except last column
y = dataset.iloc[:, -1].values # all rows, just last column - whether credit card approved or not

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom  # using a pre-implemented som model: https://github.com/JustGlowing/minisom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's'] # circle & square
colors = ['r', 'g'] # red & green
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, # to get the circle/square center or else they'll start from top left corner
         w[1] + 0.5, # same as above
         markers[y[i]], 
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)