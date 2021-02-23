from collections import Counter

import numpy as np


class FPPCA(object):
    def __init__(self, W, Sigma, P, NR_b, DS, DR, muS, muR):

        # P = dimensionality of the latent variable
        self.P = P
        # sigma2 = variance of the noise
        self.Sigma = Sigma
        # W = projection matrix DxP
        self.W = W
        # print("initial W :\n", W)
        # NR = number of R in each batch
        self.NR_b = NR_b
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
        self.XS_b = None
        # XR is NR_b*DR
        self.XR_b = None
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
            print("fit data")
            FK_b = np.array(np.squeeze(data_XS_b_FK[:, 3]))
            # print("FK_b:\n", FK_b)
            # in each batch how many tuples matching RID
            key = np.unique(FK_b)
            matching_cnt = []
            for k in key:
                # return a list of true or false
                mask = (FK_b == k)
                y_new = FK_b[mask]
                matching_cnt.append(y_new.size)
            # print("matching_cnt:\n", matching_cnt)
            self.matching_cnt = matching_cnt

            data_XS_b = data_XS_b_FK[:, 0:3]
            self.XS_b = data_XS_b
            # print('XS_b:\n', self.XS_b)
            self.XR_b = data_XR_b
            # print('XR_b:\n', self.XR_b)
            self.NS_b = data_XS_b.shape[0]
            # print('NS_b', self.NS_b)
            # print('NR_b', self.NR_b)


    def calculate_E_W(self):

        print("Calculate_E_W")
        muS = self.muS
        muR = self.muR
        P = self.P
        XS_b = self.XS_b
        XR_b = self.XR_b
        NS_b = self.NS_b
        NR_b = self.NR_b
        DS = self.DS
        DR = self.DR
        D = self.D
        GS = self.GS
        GR = self.GR
        K = self.K
        XR_b_mu = np.zeros((DR, NR_b))
        XS_b_mu = np.zeros((DS, NS_b))
        ExpZ_b_R = np.zeros((P, NR_b))
        ExpZ_b_S = np.zeros((P, NS_b))
        ExpZ_b = np.zeros((P, NS_b))
        ExpZZT_b = np.zeros((NS_b, P, P))

        U = np.zeros((NS_b, DS, P))
        L = np.zeros((NS_b, DR, P))
        W_b_p1 = np.zeros((NS_b, D, P))
        matching_cnt = self.matching_cnt

        ns = 0
        nr = 0
        for nr_m in range(NR_b):
            # print("nr:", nr)
            # DR x 1 XR_b_mu[:, [nr]] can be used repeatedly
            XR_b_mu[:, [nr]] = (XR_b[[nr], :] - muR).T
            # print("XR_b_mu[", nr, "]:\n", XR_b_mu[:, [nr]])
            # Px1  ExpZ_b_R[:, [nr]] can be used repeatedly
            ExpZ_b_R[:, [nr]] = GR.dot(XR_b_mu[:, [nr]])
            # print("ExpZ_b_R[", nr, "]:\n", ExpZ_b_R[:, [nr]])
            for ns_m in range(matching_cnt[nr]):
                # print("ns:", ns)
                # DS x 1
                XS_b_mu[:, [ns]] = (XS_b[[ns], :] - muS).T
                ExpZ_b_S[:, [ns]] = GS.dot(XS_b_mu[:, [ns]])
                # print("ExpZ_b_S[", ns, "]:\n", ExpZ_b_S[:, [ns]])
                ExpZ_b[:, [ns]] = ExpZ_b_S[:, [ns]] + ExpZ_b_R[:, [nr]]
                # print("ExpZ_b[", ns, "]:\n", ExpZ_b[:, [ns]])
                ExpZZT_b[[ns], :, :] = (K + ExpZ_b[:, [ns]].dot(ExpZ_b[:, [ns]].T))[np.newaxis, :, :]
                # print("ExpZZT_b[", ns, "]:\n", ExpZZT_b[[ns], :, :])

                U[[ns], :, :] = XS_b_mu[:, [ns]].dot(ExpZ_b[:, [ns]].T)
                L[[ns], :, :] = XR_b_mu[:, [nr]].dot(ExpZ_b[:, [ns]].T)
                W_b_p1[[ns], :, :] = np.hstack((U[[ns], :, :], L[[ns], :, :]))
                # print("W_b_p1[[", ns, "] :, :]:\n", W_b_p1[[ns], :, :])
                ns = ns + 1
            nr = nr + 1
        print(NS_b == ns)
        # print("XR_b_mu:\n", XR_b_mu, XR_b_mu.shape)
        # print("XS_b_mu:\n", XS_b_mu, XS_b_mu.shape)
        # ExpZ_b= P x NS_b
        # print("ExpZ_b:\n", ExpZ_b, ExpZ_b.shape)
        # ExpZZ_b=NS_b x P x P
        # print("ExpZZT_b:\n", ExpZZT_b, ExpZZT_b.shape)
        # W_b_p1=NS_b x D x P
        # W_b_p1_sum= D x P
        W_b_p1_sum = np.sum(W_b_p1, axis=0)
        # W_b_p2_sum= P x P
        W_b_p2_sum = np.sum(ExpZZT_b, axis=0)
        # print("W_b_p1_sum:\n", W_b_p1_sum, W_b_p1_sum.shape)
        # print("W_b_p2_sum:\n", W_b_p2_sum, W_b_p2_sum.shape)
        # print("ns:", ns)
        return W_b_p1_sum, W_b_p2_sum

    def calculate_Sigma2(self, Wnew):
        print("calculate Sigma2")
        muS = self.muS
        muR = self.muR
        P = self.P
        XS_b = self.XS_b
        XR_b = self.XR_b
        NS_b = self.NS_b
        NR_b = self.NR_b
        DS = self.DS
        DR = self.DR
        D = self.D
        matching_cnt = self.matching_cnt
        GS = self.GS
        GR = self.GR
        K = self.K
        # print("GS in calculate_Sigma2:", GS)
        XR_b_mu = np.zeros((DR, NR_b))
        XS_b_mu = np.zeros((DS, NS_b))
        ExpZ_b_R = np.zeros((P, NR_b))
        ExpZ_b_S = np.zeros((P, NS_b))
        ExpZ_b = np.zeros((P, NS_b))
        ExpZZT_b = np.zeros((NS_b, P, P))

        Sigma_b_p1_R = np.zeros((1, NR_b))
        Sigma_b_p1_S = np.zeros((1, NS_b))
        Sigma_b_p1 = np.zeros((1, NS_b))
        Wnew_S = Wnew[0:DS, :]
        Wnew_R = Wnew[DS:D, :]
        Wnew_XR_mu = np.zeros((P, NR_b))
        Wnew_XS_mu = np.zeros((P, NS_b))
        Wnew_X_mu = np.zeros((P, NS_b))
        Sigma_b_p2 = np.zeros((1, NS_b))

        WW = Wnew.T.dot(Wnew)
        Sigma_b_p3 = np.zeros((1, NS_b))
        ns = 0
        nr = 0
        for nr_m in range(NR_b):
            # print("nr:", nr)
            # DR x 1 XR_b_mu[:, [nr]] can be used repeatedly
            XR_b_mu[:, [nr]] = (XR_b[[nr], :] - muR).T
            # print("XR_b_mu[", nr, "]:\n", XR_b_mu[:, [nr]])
            # Px1  ExpZ_b_R[:, [nr]] can be used repeatedly
            ExpZ_b_R[:, [nr]] = GR.dot(XR_b_mu[:, [nr]])
            # print("ExpZ_b_R[", nr, "]:\n", ExpZ_b_R[:, [nr]])
            Sigma_b_p1_R[:, [nr]] = np.sum(np.power(XR_b_mu[:, [nr]], 2), axis=0)
            # print("Sigma_b_p1_R[", nr, "]:\n", Sigma_b_p1_R[:, [nr]])
            Wnew_XR_mu[:, [nr]] = Wnew_R.T.dot(XR_b_mu[:, [nr]])
            for ns_m in range(matching_cnt[nr]):
                # print("ns:", ns)
                # DS x 1
                XS_b_mu[:, [ns]] = (XS_b[[ns], :] - muS).T
                Sigma_b_p1_S[:, [ns]] = np.sum(np.power(XS_b_mu[:, [ns]], 2), axis=0)
                # print("Sigma_b_p1_S[", ns, "]:\n", Sigma_b_p1_S[:, [ns]])
                # not change for every iteration!!!
                Sigma_b_p1[:, [ns]] = Sigma_b_p1_S[:, [ns]] + Sigma_b_p1_R[:, [nr]]
                # print("Sigma_b_p1[", ns, "]:\n", Sigma_b_p1[:, [ns]])
                ExpZ_b_S[:, [ns]] = GS.dot(XS_b_mu[:, [ns]])
                # print("ExpZ_b_S[", ns, "]:\n", ExpZ_b_S[:, [ns]])
                ExpZ_b[:, [ns]] = ExpZ_b_S[:, [ns]] + ExpZ_b_R[:, [nr]]
                # print("ExpZ_b[", ns, "]:\n", ExpZ_b[:, [ns]])
                Wnew_XS_mu[:, [ns]] = Wnew_S.T.dot(XS_b_mu[:, [ns]])
                Wnew_X_mu[:, [ns]] = Wnew_XS_mu[:, [ns]] + Wnew_XR_mu[:, [nr]]
                Sigma_b_p2[:, [ns]] = 2 * ExpZ_b[:, [ns]].T.dot(Wnew_X_mu[:, [ns]])
                # print("Sigma_b_p2[", ns, "]:\n", Sigma_b_p2[:, [ns]])
                ExpZZT_b[[ns], :, :] = (K + ExpZ_b[:, [ns]].dot(ExpZ_b[:, [ns]].T))[np.newaxis, :, :]
                # print("ExpZZT_b[", ns, "]:\n", ExpZZT_b[[ns], :, :])
                Sigma_b_p3[:, [ns]] = np.trace(np.squeeze(ExpZZT_b[[ns], :, :]).dot(WW))

                ns = ns + 1
            nr = nr + 1
        print(NS_b == ns)
        Sigma_b_p1_sum = np.sum(Sigma_b_p1, axis=1)
        # print("Sigma_b_p1_sum", Sigma_b_p1_sum, Sigma_b_p1_sum.shape)
        Sigma_b_p2_sum = np.sum(Sigma_b_p2, axis=1)
        # print("Sigma_b_p2_sum", Sigma_b_p2_sum, Sigma_b_p2_sum.shape)
        Sigma_b_p3_sum = np.sum(Sigma_b_p3, axis=1)
        # print("Sigma_b_p3_sum", Sigma_b_p3_sum, Sigma_b_p3_sum.shape)
        return Sigma_b_p1_sum, Sigma_b_p2_sum, Sigma_b_p3_sum

