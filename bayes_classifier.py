"""
Naive means all input features are independent
    P(X|C) = mult(p(X_i|C))

Non-Naive means we just can't equate the above (no independence)
    P(X|C) = HMMs
"""

from future.utils import iteritems
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('large_files/train.csv')
    # data = df.as_matrix() # turn data into numpy ndarray
    data = df.values # turn data into numpy ndarray
    min=data[:, 1:].min()
    max=data[:, 1:].max()
    print(f"the min and max value of data is {min} and {max} accordingly")
    np.random.shuffle(data)
    print("Pixel intensities are a number in: [1,255]; We will scale to [0,1]")
    X = data[:, 1:] / float(max) # data[:, 1:] / 255.0 normalize the data so it is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

class Bayes(object):
    def fit(self, X, Y, smoothing=1e-2):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'cov': np.cov(current_x.T) + np.eye(D)*smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            mean, cov = g['mean'], g['cov']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    X, Y = get_data(10000)
    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Bayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

    # plot the mean of each class
    for c, g in iteritems(model.gaussians):
        plt.imshow(g['mean'].reshape(28, 28))
        plt.title(c)
        plt.show()