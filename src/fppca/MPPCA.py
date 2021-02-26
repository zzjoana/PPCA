import numpy as np


class MPPCA(object):
    def __init__(self, P, W, Sigma, D, mu):
        # P = dimensionality of the latent variable
        self.P = P
        # sigma2 = variance of the noise
        self.Sigma = Sigma
        # D = dimensionality of the  data
        self.D = D
        # N_b = number of data points each batch
        self.N_b = 0
        # mu = mean of the data Dx1
        self.mu = mu
        # W = projection matrix DxP
        self.W = W
        # X NxD
        self.X_b = None
        M = self.W.T.dot(self.W) + self.Sigma * np.eye(self.P)
        M_1 = np.linalg.inv(M)
        # G = M*D
        self.G = M_1.dot(self.W.T)
        self.K = self.Sigma * M_1

    # Fit the model data X= W*Z + mu + sigma2*I
    def fit(self, data_X_b):
        print("fit data")
        # initialize W to small random numbers
        self.X_b = data_X_b  # N*D
        # print("X_b:\n", self.X_b)
        # print(self.X_b.shape)
        self.N_b = data_X_b.shape[0]  # number of data points (number of column)
        # print("mu:\n", self.mu, "N_b:", self.N_b, "D:", self.D, "P:", self.P)

    def calculate_E_W(self):

        print("Calculate_E_W")
        mu = self.mu
        P = self.P
        X_b = self.X_b
        N_b = self.N_b
        D = self.D
        G = self.G
        K = self.K
        # print("mu:\n", mu)
        # print("D:\n", D)
        # print("P:\n", P)
        # print("X_b:\n", X_b)
        # print("N_b:\n", N_b)
        X_b_mu = np.zeros((D, N_b))
        ExpZ_b = np.zeros((P, N_b))
        ExpZZT_b = np.zeros((N_b, P, P))
        W_b_p1 = np.zeros((N_b, D, P))

        for n in range(N_b):
            print("n:", n)
            X_b_mu[:, [n]]=(X_b[[n], :] - mu).T
            # print("(X_b[", n, "] - mu).T:\n", (X_b[[n], :] - mu).T)
            ExpZ_b[:, [n]] = G.dot(X_b_mu[:, [n]])
            # print("ExpZ_b[", n, "]:\n", ExpZ_b[:, [n]])
            # PxP + Px1 * 1xP = P*P the Z is the first dim in the 3-dim matrix
            ExpZZT_b[[n], :, :] = (K + ExpZ_b[:, [n]].dot(ExpZ_b[:, [n]].T))[np.newaxis, :, :]
            # print("ExpZ_b[", n, "]^T:\n", ExpZ_b[:, [n]].T)
            # print("ExpZ_b[", n, "].dot(ExpZ_b[", n, "].T):\n", ExpZ_b[:, [n]].dot(ExpZ_b[:, [n]].T))
            W_b_p1[[n], :, :] = (X_b_mu[:, [n]].dot(ExpZ_b[:, [n]].T))[np.newaxis, :, :]

        # NxDxP sum
        # W_b_p1_sum= D x P
        W_b_p1_sum = np.sum(W_b_p1, axis=0)
        # W_b_p2_sum= P x P
        W_b_p2_sum = np.sum(ExpZZT_b, axis=0)
        return W_b_p1_sum, W_b_p2_sum



    def calculate_Sigma2(self, Wnew):
        print("Calculate_E_W")
        mu = self.mu
        P = self.P
        X_b = self.X_b
        N_b = self.N_b
        D = self.D
        G = self.G
        K = self.K
        X_b_mu = np.zeros((D, N_b))
        ExpZ_b = np.zeros((P, N_b))
        ExpZZT_b = np.zeros((N_b, P, P))
        Sigma_b_p1 = np.zeros((1, N_b))
        Sigma_b_p2 = np.zeros((1, N_b))
        Sigma_b_p3 = np.zeros((1, N_b))
        WW = Wnew.T.dot(Wnew)
        for n in range(N_b):
            X_b_mu[:, [n]] = (X_b[[n], :] - mu).T
            # print("(X_b[", n, "] - mu).T:\n", (X_b[[n], :] - mu).T)
            ExpZ_b[:, [n]] = G.dot(X_b_mu[:, [n]])
            # print("ExpZ_b[", n, "]:\n", ExpZ_b[:, [n]])
            # PxP + Px1 * 1xP = P*P the Z is the first dim in the 3-dim matrix
            ExpZZT_b[[n], :, :] = (K + ExpZ_b[:, [n]].dot(ExpZ_b[:, [n]].T))[np.newaxis, :, :]
            Sigma_b_p1[:, [n]] = np.sum(np.power(X_b_mu[:, [n]], 2), axis=0)
            # print("(X_b[", n, "] - mu).T:\n", (X_b[[n], :] - mu).T)
            # print("Sigma_b_p1[:,", n, "]:\n", Sigma_b_p1[:, [n]])
            Sigma_b_p2[:, [n]] = 2 * ExpZ_b[:, [n]].T.dot(Wnew.T).dot(X_b_mu[:, [n]])
            # print("Sigma_b_p2[:,", n, "]:\n", Sigma_b_p2[:, [n]])
            Sigma_b_p3[:, [n]] = np.trace(np.squeeze(ExpZZT_b[[n], :, :]).dot(WW))
            # print("before squeeze:\n", ExpZZT[[n], :, :])
            # print("after squeeze:\n", np.squeeze(ExpZZT[[n], :, :]))
            # print("sigma2_p3[:,", n, "]:\n", sigma2_p3[:, [n]])
        # print("Sigma_b_p1:\n", Sigma_b_p1)
        # print("Sigma_b_p2:\n", Sigma_b_p2)
        # print("sigma2_p3:\n", sigma2_p3)

        Sigma_b_p1_sum = np.sum(Sigma_b_p1, axis=1)
        # print("Sigma_b_p1_sum", Sigma_b_p1_sum, Sigma_b_p1_sum.shape)
        Sigma_b_p2_sum = np.sum(Sigma_b_p2, axis=1)
        # print("Sigma_b_p2_sum", Sigma_b_p2_sum, Sigma_b_p2_sum.shape)
        Sigma_b_p3_sum = np.sum(Sigma_b_p3, axis=1)
        # print("Sigma_b_p3_sum", Sigma_b_p3_sum, Sigma_b_p3_sum.shape)
        return Sigma_b_p1_sum, Sigma_b_p2_sum, Sigma_b_p3_sum

