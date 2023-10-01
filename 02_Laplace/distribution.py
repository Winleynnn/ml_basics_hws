import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # Your code here
        med = np.median(x, axis = 0)
        sum = 0
        # if(np.atleast_2d(x)):
        #     for i in range(x.shape[0]):
        #         for j in range(x.shape[1]):
        #             sum += np.abs(x[i][j]-med)
        #     res = sum/(x.shape[0]*x.shape[1])
        # else:
        #     for i in range(x.shape[0]):
        #         sum += np.abs(x[i]-med)
        #     res = sum/(x.shape[0])
        for i in range(len(x)):
            sum += np.abs(x[i] - med)
        res = sum / len(x)
        #print(res)
        return res
        ####

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        self.loc = np.median(features, axis = 0)
        #self.loc = self.mean_abs_deviation_from_median(features)
        # YOUR CODE HERE
        # self.scale = np.std(features)
        self.scale = self.mean_abs_deviation_from_median(features)
        # YOUR CODE HERE
        ####


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        return np.log(1/(2*self.scale) * np.exp(-(np.abs(values - self.loc)/self.scale)))
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
