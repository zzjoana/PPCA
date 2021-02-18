from math import ceil
import numpy as np
from src.fppca.FPPCA import FPPCA



# XR = {1: [5, 9], 2: [9, 7], 3: [6, 8], 4: [7, 5]}
#XS = {11: [5, 3], 12: [7, 1], 13: [8, 2], 14: [7, 4], 15: [6, 3], 16: [9, 2], 17: [5, 4], 18: [8, 2], 19: [7, 3],
#      20: [6, 1]}

NR = 4
NR_b = 2
batch = ceil(NR / NR_b)
print("batch", batch)
# 用SQL求muS和muR
muS = np.mat([1, 2])

muR = np.mat([3, 4, 5])

# data_XS_b 是不知道大小的
data_XS_b = np.mat([[1, 7], [2, 8], [3, 6]])
print("data_XS_b", data_XS_b)
print("shape: ", data_XS_b.shape)

data_XR_b = np.mat([[4, 6, 7], [5, 9, 8]])
weights_b = np.array([1, 2])

fppca = FPPCA(latent_dim=2, sigma2=1, max_iter=10, NR_b=NR_b, muS=muS, muR=muR)
fppca.fit(data_XS_b, data_XR_b)
