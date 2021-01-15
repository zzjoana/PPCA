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
        # X D*N
        self.X = None
        self.init = False

    # Fit the model data X= W*Z + mu + sigma2*I
    def fit(self, data):
        if not self.init:
            # mean of each row
            mean = np.mean(data, axis=0)
            self.mu = np.mat(mean)
            print("mu:\n", self.mu)
            print(self.mu.shape)
            # initialize W to small random numbers
            self.X = data  # N*D
            print("X:\n", self.X)
            print(self.X.shape)
            self.N = data.shape[0]  # number of data points (number of column)
            self.D = data.shape[1]  # number of dimensions of the data (number of row)
            print("N:", self.N, "D:", self.D, "P:", self.P)
            self.W = np.random.rand(self.D, self.P)
            print("W:", self.W)
            self.init = True

            # W and sigma2 are found by EM algorithm
            self.expectation_maximization()
        # return data

        # EM algorithm finds the model parameters W, and sigma^2

    def expectation_maximization(self):

        print("EM algorithm")
        W = self.W
        mu = self.mu
        sigma2 = self.sigma2
        P = self.P
        X = self.X
        N = self.N
        ExpZ = np.zeros((P, N))
        ExpZZT = np.zeros((P, P, N))
        W_p1 = None
        Sigma2_p1 = None
        Sigma2_p2 = None
        Sigma2_p3 = None
        for i in range(self.max_iter):
            print("iteration " + str(i))
            # E step
            # PxD * DxP + PxP = PxP
            M = W.T.dot(W) + sigma2 * np.eye(P)
            print("M:\n", M)
            M_1 = np.linalg.inv(M)
            print("M^-1:\n", M_1)
            for n in range(N):
                # matrix of E[Zn] for all N variables # calculate the expectation of latent variable Z
                # E[Z] = inv(M)*(W.T)*(x-mu)# Px1= PxP * PxD *(D*1)
                print("cha:\n", (X[n, :] - mu).T.shape)
                print("W^T:\n", W.T)
                ExpZ[:, [n]] = M_1.dot(W.T).dot((X[n, :] - mu).T)
                print("ExpZ(n):\n", ExpZ[:, [n]].shape)
                # PxP + Px1 * 1xP = P*P
                print(ExpZ[:, [n]].dot(ExpZ[:, [n]].T).shape)
                print("ExpZZT(n):\n", ExpZZT[:, :, [n]].shape)
                ExpZZT[:, :, [n]] = (sigma2 * M_1 + ExpZ[:, [n]].dot(ExpZ[:, [n]].T))[:, :, np.newaxis]
                # (Dx1 -Dx1) * 1xP = DxP
                W_p1[:, :, n] = (X[:, n] - mu).T.dot(ExpZ[:, n].T)

            # M step
            # DxP * PxP = DxP
            W_new = (np.sum(W_p1, axis=2)).dot(np.linalg.inv(np.sum(ExpZZT, axis=2)))
            for n in range(n):
                Sigma2_p1[n] = np.linalg.norm(X[:, n] - mu)
                Sigma2_p2[n] = 2 * ExpZ[:, n].T.dot(W_new).T.dot((X[:, n] - mu).T)
                Sigma2_p3[n] = np.trace(ExpZZT[:, :, n].dot(W_new).T.dot(W_new))

            Sigma2_new = (1 / (self.N * self.D)) * np.sum(Sigma2_p1 - Sigma2_p2 + Sigma2_p3, axis=0)

            W = W_new
            sigma2 = Sigma2_new
            print("W:\n", W)
            print("sigma2:\n", sigma2)
        self.W = W
        self.sigma2 = sigma2
