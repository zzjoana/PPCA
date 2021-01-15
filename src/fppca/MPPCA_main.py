import numpy as np
import math
from scipy.spatial import distance
from sklearn.model_selection import train_test_split


from src.fppca.MPPCA import MPPCA


def generate_multivariate(Dim, s, num_points):
    # generate a random N dimensional mean vector with values [0,10]
    Mean = np.random.randint(10, size=Dim)
    # sample uniform random variable
    C = np.concatenate((np.random.uniform(0, 1, Dim), np.random.uniform(0, 1, Dim)))[:, np.newaxis]
    C.shape = (Dim, 2)

    # Create the random, sparse, symmetric adjacent matrix A
    A = np.zeros((Dim, Dim))
    for i in range(Dim):
        for j in range(i + 1):
            Pr = (1 / np.sqrt(2 * math.pi)) * math.exp((-1 / (2 * s)) * distance.euclidean(C[i], C[j]))
            uniform = np.random.uniform(0, 1, 1)  # draw one sample from the uniform variable
            if Pr >= uniform:  # If the value is greater than the sample
                A[i, j] = 1  # we put an edje
                if i != j:  # replicate to the upper half
                    A[j, i] = 1
            else:  # if the value of Pr is less we put a zero
                A[i, j] = 0
                if i != j:  # replicate to the upper half
                    A[j, i] = 0
    Precision = np.zeros((Dim, Dim))
    # based on the random adjacency matrix we make a random precision matrix with the edges replaced by 0.245
    for i in range(Dim):
        for j in range(i + 1):
            if j == i:
                Precision[i, j] = 1
            else:
                if A[i, j] == 1:
                    Precision[i, j] = 0.245
                    Precision[j, i] = 0.245
    Covariance = np.linalg.inv(Precision)  # covariance is the inverse of the precision
    data = np.random.multivariate_normal(Mean, Covariance, num_points)  # generate the data based on mean and covariance
    return data


D = 5
N = 10
s = 1 / 8
# generate data by sampling from N dimensional Gaussian
data = generate_multivariate(D, s, N)
print("data:\n", data)
# split the data into training and validation sets
mppca = MPPCA( latent_dim=2,sigma2=1,max_iter=20)
mppca.fit(data)
