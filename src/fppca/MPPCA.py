import numpy as np


class MPPCA(object):
    def __init__(self, latent_dim, sigma2, max_iter):
        # P = dimensionality of the latent variable
        self.P = latent_dim
        # sigma2 = variance of the noise
        self.sigma2 = sigma2
        # D = dimensionality of the  data
        self.D = 0
        # N = number of data points
        self.N = 0
        # mu = mean of the data Dx1
        self.mu = None
        # W = projection matrix DxP
        self.W = None
        # maximum iterations to do
        self.max_iter = max_iter
        # X NxD
        self.X = None
        self.init = False

    # Fit the model data X= W*Z + mu + sigma2*I
    def fit(self, data):
        if not self.init:
            # mean of each row
            mean = np.mean(data, axis=0)
            print("dtype:", mean.dtype)
            self.mu = np.mat(mean)
            # print("mu:\n", self.mu)
            # print(self.mu.shape)
            # initialize W to small random numbers
            self.X = data  # N*D
            # print("X:\n", self.X)
            # print(self.X.shape)
            self.N = data.shape[0]  # number of data points (number of column)
            self.D = data.shape[1]  # number of dimensions of the data (number of row)
            # print("N:", self.N, "D:", self.D, "P:", self.P)
            np.random.seed(2)
            self.W = np.random.rand(self.D, self.P)
            # self.W = np.full((self.D, self.P), 0.5)

            # print("W:", self.W)
            self.init = True

            # W and sigma2 are found by EM algorithm
            self.expectation_maximization()
        return data

        # EM algorithm finds the model parameters W, and sigma^2

    def expectation_maximization(self):

        print("EM algorithm")
        mu = self.mu
        W = self.W
        sigma2 = self.sigma2
        D = self.D
        P = self.P
        X = self.X
        N = self.N
        print("mu:\n", mu)
        print("W:\n", W)
        print("sigma2:\n", sigma2)
        print("D:\n", D)
        print("P:\n", P)
        print("X:\n", X)
        print("N:\n", N)
        ExpZ = np.zeros((P, N))
        ExpZZT = np.zeros((N, P, P))
        W_p1 = np.zeros((N, D, P))
        sigma2_p1 = np.zeros((1, N))
        sigma2_p2 = np.zeros((1, N))
        sigma2_p3 = np.zeros((1, N))

        for i in range(self.max_iter):
            print("******************************************************** iteration:" + str(i))
            print("E-step")
            # PxD * DxP + PxP = PxP
            M = W.T.dot(W) + sigma2 * np.eye(P)
            # print("M:\n", M)
            M_1 = np.linalg.inv(M)
            # print("M^-1:\n", M_1)
            ExpZ_p1 = M_1.dot(W.T)
            # print("ExpZ_p1:\n", ExpZ_p1)
            ExpZZT_p1 = sigma2 * M_1
            # print("ExpZZT_p1:\n", ExpZZT_p1)
            print("calculate expectation of Z")
            for n in range(N):
                # print("(X[", n, "] - mu).T:\n", (X[[n], :] - mu).T)
                ExpZ[:, [n]] = ExpZ_p1.dot((X[[n], :] - mu).T)
                # print("ExpZ[", n, "]:\n", ExpZ[:, [n]])
                # PxP + Px1 * 1xP = P*P the Z is the first dim in the 3-dim matrix
                ExpZZT[[n], :, :] = (ExpZZT_p1 + ExpZ[:, [n]].dot(ExpZ[:, [n]].T))[np.newaxis, :, :]
                # print("ExpZ[", n, "]^T:\n", ExpZ[:, [n]].T)
                # print("ExpZ[", n, "].dot(ExpZ[", n, "].T):\n", ExpZ[:, [n]].dot(ExpZ[:, [n]].T))

            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ExpZ^T:\n", ExpZ.T)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ExpZZT:\n", ExpZZT)

            # M step
            print("M-step")
            for n in range(N):
                W_p1[[n], :, :] = ((X[[n], :] - mu).T.dot(ExpZ[:, [n]].T))[np.newaxis, :, :]
                # print("W_p1[[", n, "] :, :]:\n", W_p1[[n], :, :])

            # NxDxP sum
            W_sum1 = np.sum(W_p1, axis=0)
            # print("W_sum1:\n", W_sum1)
            W_sum2 = np.sum(ExpZZT, axis=0)
            # print("W_sum2:\n", W_sum2)
            W_new = W_sum1.dot(np.linalg.inv(W_sum2))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!W_new:\n", W_new)
            for n in range(N):
                sigma2_p1[:, [n]] = np.square(np.linalg.norm(X[[n], :] - mu))
                # print("(X[", n, "] - mu).T:\n", (X[[n], :] - mu).T)
                # print("sigma2_p1[:,", n, "]:\n", sigma2_p1[:, [n]])
                sigma2_p2[:, [n]] = 2 * ExpZ[:, [n]].T.dot(W_new.T).dot((X[[n], :] - mu).T)
                # print("sigma2_p2[:,", n, "]:\n", sigma2_p2[:, [n]])
                sigma2_p3[:, [n]] = np.trace(np.squeeze(ExpZZT[[n], :, :]).dot(W_new.T).dot(W_new))
                # print("before squeeze:\n", ExpZZT[[n], :, :])
                # print("after squeeze:\n", np.squeeze(ExpZZT[[n], :, :]))
                # print("sigma2_p3[:,", n, "]:\n", sigma2_p3[:, [n]])
            # print("sigma2_p1:\n", sigma2_p1)
            # print("sigma2_p2:\n", sigma2_p2)
            # print("sigma2_p3:\n", sigma2_p3)

            # 1xN sum according to N
            sigma2_new = (1 / (self.N * self.D)) * (np.sum(sigma2_p1) - np.sum(sigma2_p2) + np.sum(sigma2_p3))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!sigma2_new:\n", sigma2_new)

            W = W_new
            sigma2 = sigma2_new

        self.W = W
        self.sigma2 = sigma2

    def transform_data(self, data):
        P = self.P
        mu = self.mu
        W_final = self.W
        sigma2_final = self.sigma2
        M_final = W_final.T.dot(W_final) + sigma2_final * np.eye(P)
        latent_data = np.linalg.inv(M_final).dot(W_final.T).dot((data - mu).T)  # latent = inv(M)*W.T*(data-mean)
        # return PxN matrix
        return latent_data

    def inverse_constructed(self, data):
        D = self.D
        N = self.N
        mu = self.mu
        W_final = self.W
        sigma2_final = self.sigma2
        noise_mean = np.zeros(D)
        # print(noise_mean)
        noise_var = sigma2_final * np.eye(D)
        # print(noise_var)

        noise = np.random.multivariate_normal(noise_mean, noise_var, N)
        # print("W:\n", W_final)
        # print("reduced_data:\n", data)
        # print("muT:\n", mu.T)
        # print("noiseT:\n", noise.T)
        constructed_data = W_final * data + mu.T + noise.T
        return constructed_data
