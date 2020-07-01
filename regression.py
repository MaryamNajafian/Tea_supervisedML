import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

N = 200
X = np.linspace(0, 10, N).reshape(N, 1)
Y = np.sin(X)

Ntrain = 20
idx = np.random.choice(N, Ntrain)
Xtrain = X[idx]
Ytrain = Y[idx]

# it weights the neighbors by 'distance' instead
# of just averaging the neighbors

knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
knn.fit(Xtrain, Ytrain)
knn = knn.predict(X)

# because we didnt set max_depth of the tree during
# the training it overfit the training data
dt = DecisionTreeRegressor()
dt.fit(Xtrain, Ytrain)
Ydt = dt.predict(X)

plt.scatter(Xtrain, Ytrain) # show the training points
plt.plot(X, Y) # show the original data
plt.plot(X, Yknn, label='KNN')
plt.plot(X, Ydt, label='Decision Tree')
plt.legend()
plt.show()


