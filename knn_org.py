"""
* K-nearest neighbors
 This is an example of a K-Nearest Neighbors classifier on MNIST data.
 We try k=1...5 to show how we might choose the best k.
 sudo pip install sortedcontainers (if you don't have it)

 keeping track of K-closest distances: searching a sorted list O(log K) so we use sorted list
    Small K -> leads to very complex model that over-fits the training data
    Big K -> leads to less complex model that under-fits the data and the model is too smooth

X, self.X : calculate the distance of every point to itself and its neighbors and
sl : only save the top k smallest distances in a sorted list and update it in every iteration
vote_counter: for every point you have an sl sorted list, you need to pick a class for that point that occurred most freq.
y: return the predicted label for every sample in X
"""
#%%
import numpy as np
from sortedcontainers import SortedList
from util import get_data
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as plt

class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):

        """
        Loop through all data points and save its K-Nearest
        Neighbors in form of (distance,label) in a SortedList
        """
        y = np.zeros(len(X)) # we need a prediction for every input

        for i,x in enumerate(X): # test points
           # we create a sorted list for saving k neighbors for every element in X
            sl = SortedList() # stores (distance, class) tuples

            # looping through np array row-wise
            for j,xt in enumerate(self.X): # we loop through all training points and measure the  distance
                diff = x - xt
                d = diff.dot(diff) #finds squared distance , you can use Euclidean distance if you want, same result
                if len(sl) < self.k: # K-nearest neighbors only! we don't save anything beyond K distance
                    # don't need to check, just add
                    sl.add( (d, self.y[j]) )
                else:
                    if d < sl[-1][0]:# last element of the sorted list always contains the biggest distance
                        del sl[-1]
                        sl.add( (d, self.y[j]) ) #saving distance and its associated label
            # print "input:", x
            # print "sl:", sl


            # vote
            """
            Now after saving the K-nearst neighbors it's time for saving their corresponding classes
            counting the class votes for data-point i = {class1:#votes, class2:#votes, etc}
            """
            vote_counter = {}
            for _, v in sl: # looping through the sorted list of the k-nearest neighbors
                # print "v:", v, v is class label
                vote_counter[v] = vote_counter.get(v,0) + 1
            # print "votes:", votes, "true:", Ytest[i]
            max_votes = 0
            max_votes_class = -1
            for v,count in vote_counter.iteritems():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y) #accuracy sum/#


if __name__ == '__main__':
    X, Y = get_data(2000)
    Ntrain = 1000 #limit the train data
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    train_scores = []
    test_scores = []
    ks = (1,2,3,4,5)
    for k in ks:
        print("\nk =", k)
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print("Training time:", (datetime.now() - t0))

        t0 = datetime.now()
        train_score = knn.score(Xtrain, Ytrain)
        train_scores.append(train_score)
        print("Train accuracy:", train_score)
        print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

        t0 = datetime.now()
        test_score = knn.score(Xtest, Ytest)
        print("Test accuracy:", test_score)
        test_scores.append(test_score)
        print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))

    plt.plot(ks, train_scores, label='train scores')
    plt.plot(ks, test_scores, label='test scores')
    plt.legend()
    plt.show()



