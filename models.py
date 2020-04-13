""" 
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np

class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures


    def fit(self, *, X, iterations):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        raise NotImplementedError()


    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class LambdaMeans(Model):

    def __init__(self, *, nfeatures, lambda0):
        super().__init__(nfeatures)
        """
        Args:
            nfeatures: size of feature space (only needed for _fix_test_feats)
            lambda0: A float giving the default value for lambda
            mu_k: vector of cluster means, the size of this vector will change
        """
        self.lambda0 = lambda0
        self.mu_k = None
        # TODO: Initializations etc. go here.            


    def fit(self, *, X, iterations):
        """
        Fit the LambdaMeans model.
        Note: labels are not used here.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        # TODO: Implement this!
        self.mu_k = []
        mean = np.asfarray(X.mean(axis=0))[0] #center of first clsuter is mean of the data
        self.mu_k.append(mean)

        my_iterations = 2
        my_iterations = iterations

        (n, num_features) = X.shape
        if(self.lambda0 == 0):
            #calculate lambda 0 to be the standard deviation of the points
            totalDistanceFromMean = 0
            for x_i in X:
                x_i = x_i.toarray()[0]
                totalDistanceFromMean+=self.distance(x_i, mean)
            self.lambda0 = totalDistanceFromMean/n
        print('lambda0', self.lambda0)
        clusterBins = [] #each cluster has a bin of points
        clusterBins.append([]) #since we start with 1 cluster
        for iteration in range(my_iterations):
            #clear the assignments to those clusters, we will reassign them now
            for i in range(len(self.mu_k)):
                clusterBins[i] = []
            # for each point, put it in a cluster bin
            for i, x_i in enumerate(X):
                x_i = x_i.toarray()[0]
                min_distance = self.distance(x_i, self.mu_k[0])
                min_center_index = 0
                cur_distance = 0
                for center_index, center in enumerate(self.mu_k):
                    cur_distance = self.distance(x_i, center)
                    if cur_distance < min_distance:
                        #new best cluster
                        min_distance = cur_distance
                        min_center_index = center_index
                if(min_distance > self.lambda0):
                    # all of the clusters were bad
                    min_distance = 0
                    min_center_index = len(self.mu_k)
                    #make a new cluster, with a center at this point that was far from other clusters
                    self.mu_k.append(x_i)
                    #we have a new cluster, so add a new cluster bin
                    clusterBins.append([])
                clusterBins[min_center_index].append(x_i) #actually put this point in a cluster
            
            #M step
            # print('--------------------------number of clusters', len(self.mu_k),'--------------------------')
            for cluster_index, cluster_points in enumerate(clusterBins):
                # print(len(cluster_points), "points in cluster", cluster_index)
                # re assign the center to be the mean of the points in this cluster
                self.mu_k[cluster_index] = np.mean(cluster_points, axis=0)
        return
    def distance(self, point1, point2):
        return np.sum(((point1 - point2)**2))**(1/2)
    
    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        X = self._fix_test_feats(X)
        (n, num_features) = X.shape
        y = [0] * n
        for i, x_i in enumerate(X):
                x_i = x_i.toarray()[0]
                min_distance = self.distance(x_i, self.mu_k[0])
                min_center_index = 0
                cur_distance = 0
                for center_index, center in enumerate(self.mu_k):
                    cur_distance = self.distance(x_i, center)
                    if cur_distance < min_distance:
                        #new best cluster
                        min_distance = cur_distance
                        min_center_index = center_index
                y[i]= min_center_index #actually put this point in a cluster
        return y

class StochasticKMeans(Model):

    def __init__(self, *, nfeatures, num_clusters):
        """
        Args:
            nfeatures: size of feature space (only needed for _fix_test_feats)
            num_clusters: int giving number of clusters K
            mu_k: vector of cluster means
        """
        super().__init__(nfeatures)
        self.num_clusters = num_clusters
        self.mu_k = np.zeros((num_clusters, nfeatures))
        # TODO: Initializations etc. go here.


    def initialize_clusters(self, X):
        """
        Helper function to initialize cluster centers as described in Section 3.1.
        Call this function from fit.
        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")


    def fit(self, *, X, iterations):
        """
        Fit the LambdaMeans model.
        Note: labels are not used here.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")


    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        raise Exception("You must implement this method!")
