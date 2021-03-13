import numpy as np
import time


class SPPCA(object):
    def __init__(self, P, W, Sigma, D, mu):
        start_int = time.time()
        # P = dimensionality of the latent variable
        self.P = P
        # sigma2 = variance of the noise
        self.Sigma = Sigma
        # D = dimensionality of the  data
        self.D = D
        # N_b = number of data points each batch
        self.N_b = 0
        self.NR_b = 0
        # mu = mean of the data Dx1
        self.mu = mu
        # W = projection matrix DxP
        self.W = W
        # X NxD
        self.X_b = None
        self.matching_cnt = None
        M = self.W.T.dot(self.W) + self.Sigma * np.eye(self.P)
        M_1 = np.linalg.inv(M)
        # G = M*D
        self.G = M_1.dot(self.W.T)
        self.K = self.Sigma * M_1
        end_int = time.time()
        # print("int:", end_int - start_int)
    # Fit the model data X= W*Z + mu + sigma2*I

    def fit(self, data_X_b_RID):
        start_fit = time.time()
        RID = np.array(np.squeeze(data_X_b_RID[:, 2]))
        # print("RID:\n", RID)
        # in each batch how many tuples matching one RID order
        key = np.unique(RID)
        matching_cnt = []
        for k in key:
            # return a list of true or false
            mask = (RID == k)
            y_new = RID[mask]
            matching_cnt.append(y_new.size)
        # print("matching_cnt:\n", matching_cnt)
        self.matching_cnt = matching_cnt
        data_X_b = np.delete(data_X_b_RID, 2, axis=1)
        self.X_b = data_X_b  # N*D
        # print("X_b:\n", self.X_b)
        # print(self.X_b.shape)
        self.NR_b = len(key)
        # print("self.NR_b", self.NR_b)
        self.N_b = data_X_b.shape[0]  # number of data points (number of column)
        # print("mu:\n", self.mu, "N_b:", self.N_b, "D:", self.D, "P:", self.P)
        end_fit = time.time()
        # print("fit time:", end_fit - start_fit)

    def calculate_E_W(self):
        start_cal_W = time.time()
        mu = self.mu
        P = self.P
        X_b = self.X_b
        N_b = self.N_b
        NR_b = self.NR_b
        D = self.D
        G = self.G
        K = self.K
        matching_cnt = self.matching_cnt
        W_b_p1_sum = np.zeros((D, P))
        W_b_p2_sum = np.zeros((P, P))

        W_b_p1 = []
        W_b_p2 = []
        n = 0
        for nr in range(NR_b):
            X_b_mu = (X_b[n:n + matching_cnt[nr], :] - mu).T
            ExpZ_b = G.dot(X_b_mu)
            ExpZZT_b = (matching_cnt[nr] * K + ExpZ_b.dot(ExpZ_b.T))
            W_b_p1.append(X_b_mu.dot(ExpZ_b.T))
            W_b_p2.append(ExpZZT_b)
            n = n + matching_cnt[nr]

        for e in range(NR_b):
            W_b_p1_sum = W_b_p1_sum + W_b_p1[e]
        for e in range(NR_b):
            W_b_p2_sum = W_b_p2_sum + W_b_p2[e]
        end_cal_W = time.time()
        # print("cal_W_time:", end_cal_W-start_cal_W)
        return W_b_p1_sum, W_b_p2_sum



    def calculate_Sigma2(self, Wnew):
        start_cal_Sigma = time.time()
        mu = self.mu
        P = self.P
        X_b = self.X_b
        N_b = self.N_b
        NR_b = self.NR_b

        D = self.D
        G = self.G
        K = self.K
        matching_cnt = self.matching_cnt

        Sigma_b_p1 = []
        Sigma_b_p2 = []
        Sigma_b_p3 = []
        WW = Wnew.T.dot(Wnew)
        Sigma_b_p1_sum = 0
        Sigma_b_p2_sum = 0
        Sigma_b_p3_sum = 0
        n = 0
        for nr in range(NR_b):

            X_b_mu = (X_b[n:n + matching_cnt[nr], :] - mu).T
            # print("(X_b[", n, "] - mu).T:\n", (X_b[[n], :] - mu).T)
            ExpZ_b = G.dot(X_b_mu)
            # print("ExpZ_b[", n, "]:\n", ExpZ_b[:, [n]])
            # PxP + Px1 * 1xP = P*P the Z is the first dim in the 3-dim matrix
            ExpZZT_b = (matching_cnt[nr] * K + ExpZ_b.dot(ExpZ_b.T))
            Sigma_b_p1_matrix = np.sum(np.power(X_b_mu, 2), axis=0)
            Sigma_b_p1.append(np.sum(Sigma_b_p1_matrix))
            # print("(X_b[", n, "] - mu).T:\n", (X_b[[n], :] - mu).T)
            # print("Sigma_b_p1[:,", n, "]:\n", Sigma_b_p1[:, [n]])

            sum = 0
            start2 = time.time()
            # print("matching_cnt[nr]:", matching_cnt[nr])
            for m in range(matching_cnt[nr]):

                p2 = 2 * ExpZ_b.T[m, :].dot(Wnew.T.dot(X_b_mu)[:, m])
                # print("np.sum(p2)", np.sum(p2))
                sum = sum + np.sum(p2)

            Sigma_b_p2.append(sum)
            # Sigma_b_p2.append(2 * np.trace(ExpZ_b.T.dot(Wnew.T).dot(X_b_mu)))
            end2 = time.time()
            # print("2:", end2 - start2)

            # print("Sigma_b_p2[:,", n, "]:\n", Sigma_b_p2[:, [n]])
            Sigma_b_p3.append(np.trace(np.squeeze(ExpZZT_b).dot(WW)))
            # print("before squeeze:\n", ExpZZT[[n], :, :])
            # print("after squeeze:\n", np.squeeze(ExpZZT[[n], :, :]))
            # print("sigma2_p3[:,", n, "]:\n", sigma2_p3[:, [n]])
            n = n + matching_cnt[nr]
        # print("Sigma_b_p1:\n", Sigma_b_p1)
        # print("Sigma_b_p2:\n", Sigma_b_p2)
        # print("sigma2_p3:\n", sigma2_p3)

        for e in range(NR_b):
            Sigma_b_p1_sum = Sigma_b_p1_sum + Sigma_b_p1[e]
        for e in range(NR_b):
            Sigma_b_p2_sum = Sigma_b_p2_sum + Sigma_b_p2[e]
        for e in range(NR_b):
            Sigma_b_p3_sum = Sigma_b_p3_sum + Sigma_b_p3[e]
        end_cal_Sigma = time.time()
        # print("cal_Sigma_time:", end_cal_Sigma - start_cal_Sigma)
        return Sigma_b_p1_sum, Sigma_b_p2_sum, Sigma_b_p3_sum
