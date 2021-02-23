from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

from src.fppca.MPPCA_init import MPPCA_IN

iris_data = load_iris()
x_data = iris_data.data
y_data = iris_data.target
# data is 150*4
N = x_data.shape[0]
print('N:', N)
print(x_data.shape)

mppca = MPPCA_IN(P=2, Sigma=1, max_iter=20)
mppca.fit(x_data)
data_reduced = mppca.transform_data(x_data)
data_reduced_T = data_reduced.T
print("original data:\n", x_data)
print("reduced data^T:\n", data_reduced_T)

X_data_reduced, Y_data_reduced = data_reduced_T[:, 0], data_reduced_T[:, 1]
# print("X_data_reduced:\n", X_data_reduced)
# print("Y_data_reduced:\n", Y_data_reduced)
x_list = list(np.array(X_data_reduced).flatten())
y_list = list(np.array(Y_data_reduced).flatten())
# print("x_list:\n", x_list)
# print("y_list:\n", y_list)
plt.scatter(x_list, y_list, c='y')
# plt.show()

print(x_data.shape)
print("x_shape:", data_reduced_T.shape)
print("y_shape:", y_data.shape)
print(data_reduced_T[0, 0])
print(data_reduced_T[0, 1])
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(N):
    if y_data[i] == 0:
        red_x.append(data_reduced_T[i, 0])
        red_y.append(data_reduced_T[i, 1])
    elif y_data[i] == 1:
        blue_x.append(data_reduced_T[i, 0])
        blue_y.append(data_reduced_T[i, 1])
    else:
        green_x.append(data_reduced_T[i, 0])
        green_y.append(data_reduced_T[i, 1])
plt.scatter(red_x, red_y, c='r')
plt.scatter(blue_x, blue_y, c='b')
plt.scatter(green_x, green_y, c='g')
plt.title("N:" + str(N) + " || max_iter:" + str(mppca.max_iter) + " || sigma_seed: 2")
plt.show()

