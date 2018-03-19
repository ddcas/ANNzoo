import numpy as np

class Dataset():

    def __init__(self,N,d,z,c,labels,sd=1,mean_centre=0,mean_sd=1):
        means = []
        means.append(np.random.normal(mean_centre,mean_sd,d))
        # distribute the means so that the input data is linearly separable
        for i in range(c-1):
            means.append([mean_sd,mean_sd]-means[i])
        covs = (sd**2)*np.eye(d,d)
        # generate patterns and targets
        self.points = [np.random.multivariate_normal(means[i],covs,N//2) for i in range(c)]
        X = np.vstack(self.points)
        self.XT = np.hstack((X,labels.T))
        # shuffle input and output patterns and targets together
        np.random.shuffle(self.XT)
        self.patterns = np.ones((d+1,N))
        self.patterns[1:,:] = np.copy(self.XT[:,:-z].T)
        # self.patterns[:-1,:] = np.copy(self.XT[:,:-z].T)
        self.targets = np.copy(self.XT[:,-z:].T)
