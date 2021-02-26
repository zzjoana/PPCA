from collections import Counter

import numpy as np


class FPPCA(object):
    def __init__(self, W, Sigma, P, DS, DR, muS, muR):

        # P = dimensionality of the latent variable
        self.P = P
        # sigma2 = variance of the noise
        self.Sigma = Sigma
        # W = projection matrix DxP
        self.W = W
        # print("initial W :\n", W)
        # NR = number of R in each batch
        self.NR_b = 0
        # NS = number of S in this batch
        self.NS_b = 0

        # DS = dimensionality of S
        self.DS = DS
        # DR = dimensionality of R
        self.DR = DR
        # D = dimensionality of S+R
        self.D = self.DS + self.DR
        # muS = mean of S
        self.muS = muS
        # muR = mean of R
        self.muR = muR
        # XS is NS_b*DS
        # XR is NR_b*DR
        self.XR_b = None
        self.XS_b = None
        # matching count is a list
        self.matching_cnt = None
        # M = PxP
        M = self.W.T.dot(self.W) + self.Sigma * np.eye(self.P)
        M_1 = np.linalg.inv(M)
        # G = M*D
        G = M_1.dot(self.W.T)
        # print("G:\n", G)
        # GS = PxDS
        self.GS = G[:, 0:self.DS]
        # print("self.GS", self.GS)
        # GR = PxDR
        self.GR = G[:, self.DS:self.D]
        # print("self.GR", self.GR)
        # K= PxP
        self.K = self.Sigma * M_1
        # print("self.K", self.K)

    # fit the variables need data
    def fit(self, data_XS_b_FK, data_XR_b):
        FK_b = np.array(np.squeeze(data_XS_b_FK[:, 4]))
        # print("FK_b:\n", FK_b)
        # in each batch how many tuples matching one RID order
        key = np.unique(FK_b)
        matching_cnt = []
        for k in key:
            # return a list of true or false
            mask = (FK_b == k)
            y_new = FK_b[mask]
            matching_cnt.append(y_new.size)
        # print("matching_cnt:\n", matching_cnt)
        self.matching_cnt = matching_cnt

        self.XS_b = data_XS_b_FK[:, 0:4]
        # print('XS_b:\n', self.XS_b)
        self.XR_b = data_XR_b
        # print('XR_b:\n', self.XR_b)
        self.NR_b = data_XR_b.shape[0]
        self.NS_b = data_XS_b_FK.shape[0]

    def calculate_E_W(self):

        # print("Calculate_E_W")

        muS = self.muS
        # print("muS", muS)
        muR = self.muR
        # print("muR",muR)
        P = self.P
        XR_b = self.XR_b
        # print("XR_b",XR_b)
        XS_b = self.XS_b
        NR_b = self.NR_b
        # print("NR_b",NR_b)
        DS = self.DS
        DR = self.DR
        D = self.D
        GS = self.GS
        GR = self.GR
        K = self.K
        matching_cnt = self.matching_cnt
        XR_b_mu = np.zeros((DR, NR_b))
        ExpZ_b_R = np.zeros((P, NR_b))
        W_b_p1_sum = np.zeros((D, P))
        W_b_p2_sum = np.zeros((P, P))

        W_b_p1 = []
        W_b_p2 = []
        s = 0
        for nr in range(NR_b):
            # print("nr:", nr)
            # print("XS_b_list[nr]:\n", XS_b_list[nr], XS_b_list[nr].shape)
            # DR x1
            XR_b_mu[:, [nr]] = (XR_b[[nr], :] - muR).T
            # DS x NSi
            XS_b_mu = (XS_b[s:s + matching_cnt[nr], :] - muS).T
            # Px1
            ExpZ_b_R[:, [nr]] = GR.dot(XR_b_mu[:, [nr]])
            # Px NSi
            ExpZ_b_S = GS.dot(XS_b_mu)
            # Px NSi
            ExpZ_b = ExpZ_b_S + ExpZ_b_R[:, [nr]]
            # print("ExpZ_b:\n", ExpZ_b)
            # PxP
            ExpZZT_b = matching_cnt[nr] * K + ExpZ_b.dot(ExpZ_b.T)
            # print("ExpZZT_b:\n", ExpZZT_b)
            # DS x P
            U = XS_b_mu.dot(ExpZ_b.T)
            # DR xP
            L = XR_b_mu[:, [nr]].repeat([ExpZ_b.shape[1]], axis=1).dot(ExpZ_b.T)
            W_b_p1.append(np.vstack((U, L)))
            W_b_p2.append(ExpZZT_b)
            s = s + matching_cnt[nr]
        # print(" W_b_p2:\n", W_b_p2)

        for e in range(NR_b):
            # print("W_b_p1[e]:\n", W_b_p1[e], W_b_p1[e].shape )
            W_b_p1_sum = W_b_p1_sum + W_b_p1[e]
        for e in range(NR_b):
            W_b_p2_sum = W_b_p2_sum + W_b_p2[e]
        # print("W_b_p1_sum:\n", W_b_p1_sum)
        # print("W_b_p2_sum:\n", W_b_p2_sum)

        return W_b_p1_sum, W_b_p2_sum

    def calculate_Sigma2(self, Wnew):
        # print("calculate Sigma2")
        muS = self.muS
        muR = self.muR
        P = self.P
        XR_b = self.XR_b
        XS_b = self.XS_b
        NR_b = self.NR_b
        DS = self.DS
        DR = self.DR
        D = self.D
        GS = self.GS
        GR = self.GR
        K = self.K
        matching_cnt = self.matching_cnt
        XR_b_mu = np.zeros((DR, NR_b))
        ExpZ_b_R = np.zeros((P, NR_b))
        Sigma_b_p1_R = np.zeros((1, NR_b))
        Wnew_XR_mu = np.zeros((P, NR_b))
        # DSxP
        Wnew_S = Wnew[0:DS, :]
        # DRxP
        Wnew_R = Wnew[DS:D, :]
        Sigma_b_p1 = []
        Sigma_b_p2 = []
        Sigma_b_p3 = []
        # PxP
        WW = Wnew.T.dot(Wnew)
        Sigma_b_p1_sum = 0
        Sigma_b_p2_sum = 0
        Sigma_b_p3_sum = 0
        s = 0
        for nr in range(NR_b):
            # DR x1
            XR_b_mu[:, [nr]] = (XR_b[[nr], :] - muR).T
            # DS x NSi
            XS_b_list_mu = (XS_b[s:s + matching_cnt[nr], :] - muS).T
            # Px1
            ExpZ_b_R[:, [nr]] = GR.dot(XR_b_mu[:, [nr]])
            # Px NSi
            ExpZ_b_S = GS.dot(XS_b_list_mu)
            # Px NSi
            ExpZ_b = ExpZ_b_S + ExpZ_b_R[:, [nr]]
            # PxP
            ExpZZT_b = matching_cnt[nr] * K + ExpZ_b.dot(ExpZ_b.T)

            # 1X1
            Sigma_b_p1_R[:, [nr]] = np.sum(np.power(XR_b_mu[:, [nr]], 2), axis=0)
            # 1xNSi
            Sigma_b_p1_S = np.sum(np.power(XS_b_list_mu, 2), axis=0)
            # each element is 1xNSi
            # Sigma_b_p1_matrix=np.sum((Sigma_b_p1_R[:, [nr]] + Sigma_b_p1_S),axis=1)
            Sigma_b_p1_matrix = Sigma_b_p1_R[:, [nr]] + Sigma_b_p1_S
            Sigma_b_p1_matrix_sum = np.sum(Sigma_b_p1_matrix)
            # print("Sigma_b_p1_matrix_sum", Sigma_b_p1_matrix_sum)
            Sigma_b_p1.append(Sigma_b_p1_matrix_sum)
            # print("Sigma_b_p1", Sigma_b_p1)
            # Px1
            Wnew_XR_mu[:, [nr]] = Wnew_R.T.dot(XR_b_mu[:, [nr]])
            # PxNSi
            Wnew_XS_mu = Wnew_S.T.dot(XS_b_list_mu)
            # PxNSi
            Wnew_X_mu = Wnew_XR_mu[:, [nr]] + Wnew_XS_mu

            Sigma_b_p2.append(2 * np.trace(ExpZ_b.T.dot(Wnew_X_mu)))
            # print("Sigma_b_p2", Sigma_b_p2)
            Sigma_b_p3.append(np.trace(ExpZZT_b.dot(WW)))
            # print("Sigma_b_p3", Sigma_b_p3)
            s = s + matching_cnt[nr]
        for e in range(NR_b):
            Sigma_b_p1_sum = Sigma_b_p1_sum + Sigma_b_p1[e]
        for e in range(NR_b):
            Sigma_b_p2_sum = Sigma_b_p2_sum + Sigma_b_p2[e]
        for e in range(NR_b):
            Sigma_b_p3_sum = Sigma_b_p3_sum + Sigma_b_p3[e]
        print("Sigma_b_p1_sum", Sigma_b_p1_sum, Sigma_b_p1_sum.shape)
        print("Sigma_b_p2_sum", Sigma_b_p2_sum, Sigma_b_p2_sum.shape)
        print("Sigma_b_p3_sum", Sigma_b_p3_sum, Sigma_b_p3_sum.shape)
        return Sigma_b_p1_sum, Sigma_b_p2_sum, Sigma_b_p3_sum
