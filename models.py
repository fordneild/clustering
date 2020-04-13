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
        origin = [0] * num_features
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
                if len(clusterBins[i]) == 0:
                    self.mu_k[i] = origin
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
        # raise Exception("You must implement this method!")
        (n, num_features) = X.shape
        origin = [0] * num_features
        if len(self.mu_k) == 1:
            #make one cluter as mean of points
            self.mu_k[0] = np.asfarray(X.mean(axis=0))[0]
        else:
            #place two centers at the max and min points
            #min initialization
            min_x_i = X[0].toarray()[0]
            min_x_i_distance = self.distance(origin, X[0].toarray()[0])
            #min initialization
            max_x_i = min_x_i
            max_x_i_distance = min_x_i_distance
            for x_i in X:
                x_i = x_i.toarray()[0]
                x_i_distance = self.distance(x_i, origin)
                if( x_i_distance < min_x_i_distance):
                    min_x_i_distance = x_i_distance
                    min_x_i = x_i
                if(x_i_distance > max_x_i_distance):
                    max_x_i_distance =x_i_distance
                    max_x_i = x_i
            self.mu_k[0] = min_x_i
            self.mu_k[-1] = max_x_i
            for i in range(1, len(self.mu_k)-1):
                # for each remaining cluster, spread them out evenly
                self.mu_k[i] = min_x_i + i*(max_x_i- min_x_i)/(len(self.mu_k) - 1)
        

    def distance(self, point1, point2):
        return np.sum(((point1 - point2)**2))**(1/2)
    
    def fit(self, *, X, iterations):
        """
        Fit the LambdaMeans model.
        Note: labels are not used here.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """
        self.initialize_clusters(X)
        c = 2
        (n, num_features) = X.shape
        # TODO: Implement this!
        # raise Exception("You must implement this method!")
        clusterBins = [] #each cluster has a bin of points
        for i in range(len(self.mu_k)):
            clusterBins.append([]) #since we start with 1 cluster

        for iteration in range(iterations):
            #clear the assignments to those clusters, we will reassign them now
            beta = (iteration+1)*c #TODO: verify if should add 1
            for i in range(len(self.mu_k)):
                if len(clusterBins[i]) == 0:
                    self.mu_k[i] = [0] * num_features
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
                clusterBins[min_center_index].append(x_i) #actually put this point in a cluster
            print('--------------------------number of clusters', len(self.mu_k),'--------------------------')
            for cluster_index, cluster_points in enumerate(clusterBins):
                running_total_top = [0] * num_features
                running_total_bottom = 0
                for i, x_i in enumerate(X):
                    x_i = x_i.toarray()[0]
                    # for this point find average distance to each cluster
                    total_distance_to_clusters = 0
                    for center in self.mu_k:
                        total_distance_to_clusters+= self.distance(x_i, center)
                    d_hat = total_distance_to_clusters/len(self.mu_k)
                    top = np.exp((-beta *self.distance(x_i, self.mu_k[cluster_index]))/d_hat)
                    bottom = 0
                    for center in self.mu_k:
                        bottom+= np.exp((-beta *self.distance(x_i, center))/d_hat)
                    p_ik = top/bottom
                    running_total_top+=p_ik*x_i
                    running_total_bottom+=p_ik
                self.mu_k[cluster_index] = running_total_top/running_total_bottom

                print(len(cluster_points), "points in cluster", cluster_index)
                # re assign the center to be the mean of the points in this cluster
                # self.mu_k[cluster_index] = np.mean(cluster_points, axis=0)


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