import numpy as np
import math
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.fppca.MPPCA_init import MPPCA_IN


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
                A[i, j] = 1  # we put 1
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


# generated dataset
D = 3
N = 5
s = 1 / 8
# generate data by sampling from N dimensional Gaussian
data = generate_multivariate(D, s, N)
# data is N*D
print("data_orig:\n", data)


mppca = MPPCA_IN(P=2, Sigma=1, max_iter=50)
mppca.fit(data)
print("W_final:\n", mppca.W)
print("sigma2_final:\n", mppca.Sigma)
# data_reduced: PXN
data_reduced = mppca.transform_data(data)
data_reduced_T = data_reduced.T
print("data_reduced^T:\n", data_reduced.T)

X_data_reduced, Y_data_reduced = data_reduced_T[:, 0], data_reduced_T[:, 1]
print("X_data_reduced:\n", X_data_reduced)
print("Y_data_reduced:\n", Y_data_reduced)
x_list = list(np.array(X_data_reduced).flatten())
y_list = list(np.array(Y_data_reduced).flatten())
print("x_list:\n", x_list)
print("y_list:\n", y_list)
plt.scatter(x_list, y_list, c='g')
plt.title("N:" + str(N) + " || max_iter:" + str(mppca.max_iter) + " || sigma_seed: 2")
plt.show()

data_reconstructed = mppca.inverse_constructed(data_reduced)
print("data_reconstructed^T:\n", data_reconstructed.T)

data_combine = np.hstack((data.T, data_reconstructed))
X_data_combine, Y_data_combine, Z_data_combine = data_combine[0], data_combine[1], data_combine[2]
print("X_data_combine:\n", X_data_combine)
print("Y_data_combine:\n", Y_data_combine)
print("Z_data_combine:\n", Z_data_combine)
x_list = list(np.array(X_data_combine).flatten())
y_list = list(np.array(Y_data_combine).flatten())
z_list = list(np.array(Z_data_combine).flatten())

# print("x_list:\n", x_list)
# print("y_list:\n", y_list)
# print("z_list:\n", z_list)
print(len(x_list))

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_list[0:N], y_list[0:N], z_list[0:N], c='r')  # 绘制数据点
ax.scatter(x_list[N:2 * N], y_list[N:2 * N], z_list[N:2 * N], c='b')

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.title("N:" + str(N) + " || max_iter:" + str(mppca.max_iter) + " || sigma_seed: 2")
# for i in range(len(x_list)):
#    ax.text(x_list[i],y_list[i],z_list[i],i)

plt.show()
