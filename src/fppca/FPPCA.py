from collections import Counter

import numpy as np


class FPPCA(object):
    def __init__(self, W, sigma2, P, NR_b, DS, DR, muS, muR):
        # P = dimensionality of the latent variable
        self.P = P
        # sigma2 = variance of the noise
        self.sigma2 = sigma2
        # W = projection matrix DxP
        self.W = W
        # NR = number of R in each batch
        self.NR_b = NR_b
        # NS = number of S in this batch
        self.NS_b = 0
        # D = dimensionality of S+R
        self.D = 0
        # DS = dimensionality of S
        self.DS = DS
        # DR = dimensionality of R
        self.DR = DR
        # muS = mean of S
        self.muS = muS
        # muR = mean of R
        self.muR = muR

        # maximum iterations to do
        # XS is NS_b*DS
        self.XS_b = None
        # XR is NR_b*DR
        self.XR_b = None
        self.init = False
        self.matching_cnt = None

    #    def mu_calculation(self):

    def fit(self, data_XS_b_FK, data_XR_b):
        if not self.init:
            print("fit data")
            print('muS', self.muS)
            print('muR', self.muR)
            FK_b = np.array(np.squeeze(data_XS_b_FK[:, 3]))
            print("FK_b:\n", FK_b)
            # in each batch how many tuples matching RID
            key = np.unique(FK_b)
            # matching_cnt = {}
            # for k in key:
            #     mask = (FK_b == k)
            #     y_new = FK_b[mask]
            #     v = y_new.size
            #     matching_cnt[k] = v
            # print("matching_cnt:\n", matching_cnt)
            matching_cnt = []
            for k in key:
                # return a list of true or false
                mask = (FK_b == k)
                y_new = FK_b[mask]
                matching_cnt.append(y_new.size)
            print("matching_cnt:\n", matching_cnt)
            self.matching_cnt = matching_cnt

            data_XS_b = data_XS_b_FK[:, 0:3]
            self.XS_b = data_XS_b
            print('XS_b:\n', self.XS_b)
            self.XR_b = data_XR_b
            print('XR_b:\n', self.XR_b)
            self.NS_b = data_XS_b.shape[0]
            print('NS_b', self.NS_b)
            print('NR_b', self.NR_b)
            self.D = self.DS + self.DR
            print('D', self.D)

            self.init = True

    def calculate_E_W(self):

        print("Calculate_E_W")
        muS = self.muS
        muR = self.muR
        W = self.W
        print("initial W :\n", W)
        sigma2 = self.sigma2
        P = self.P
        XS_b = self.XS_b
        XR_b = self.XR_b
        NS_b = self.NS_b
        NR_b = self.NR_b
        DS = self.DS
        DR = self.DR
        D = self.D
        XR_b_mu = np.zeros((DR, NR_b))
        XS_b_mu = np.zeros((DS, NS_b))
        ExpZ_b_R = np.zeros((P, NR_b))
        ExpZ_b_S = np.zeros((P, NS_b))
        ExpZ_b = np.zeros((P, NS_b))
        # ExpZZT_b_R = np.zeros((NR_b, P, P))
        ExpZZT_b = np.zeros((NS_b, P, P))

        # UL = np.zeros((NS_b, DS, P))
        # UR = np.zeros((NS_b, DS, P))
        U = np.zeros((NS_b, DS, P))
        # LL = np.zeros((NS_b, DR, P))
        # LR = np.zeros((NR_b, DR, P))
        L = np.zeros((NS_b, DR, P))
        W_b_p1 = np.zeros((NS_b, D, P))
        matching_cnt = self.matching_cnt

        Sigma_b_p1_R = np.zeros((1, NR_b))
        Sigma_b_p1_S = np.zeros((1, NS_b))
        Sigma_b_p1 = np.zeros((1, NS_b))
        # M = PxP
        M = W.T.dot(W) + sigma2 * np.eye(P)
        M_1 = np.linalg.inv(M)
        # G = M*D
        G = M_1.dot(W.T)
        # print("G:\n", G)
        # GS = PxDS
        GS = G[:, 0:DS]
        print("GS", GS)
        # GR = PxDR
        GR = G[:, DS:D]
        print("GR", GR)
        # K= PxP
        K = sigma2 * M_1
        print("K", K)
        ns = 0
        nr = 0
        for nr_m in range(NR_b):
            print("nr:", nr)
            # DR x 1 XR_b_mu[:, [nr]] can be used repeatedly
            XR_b_mu[:, [nr]] = (XR_b[[nr], :] - muR).T
            print("XR_b_mu[", nr, "]:\n", XR_b_mu[:, [nr]])
            # Px1  ExpZ_b_R[:, [nr]] can be used repeatedly
            ExpZ_b_R[:, [nr]] = GR.dot(XR_b_mu[:, [nr]])
            print("ExpZ_b_R[", nr, "]:\n", ExpZ_b_R[:, [nr]])
            # P x P x 1  ExpZZT_b_R[[nr], :, :]
            # ExpZZT_b_R[[nr], :, :] = ExpZ_b_R[:, [nr]].dot(ExpZ_b_R[:, [nr]].T)
            # DR x P x 1
            # LR[[nr], :, :] = XR_b_mu[:, [nr]].dot(ExpZ_b_R[:, [nr]].T)
            # 1 x 1 Sigma_b_p1_R[:, [nr]] can be used repeatedly
            Sigma_b_p1_R[:, [nr]] = np.sum(np.power(XR_b_mu[:, [nr]], 2), axis=0)
            print("Sigma_b_p1_R[", nr, "]:\n", Sigma_b_p1_R[:, [nr]])
            for ns_m in range(matching_cnt[nr]):
                print("ns:", ns)
                # DS x 1
                XS_b_mu[:, [ns]] = (XS_b[[ns], :] - muS).T
                ExpZ_b_S[:, [ns]] = GS.dot(XS_b_mu[:, [ns]])
                print("ExpZ_b_S[", ns, "]:\n", ExpZ_b_S[:, [ns]])
                ExpZ_b[:, [ns]] = ExpZ_b_S[:, [ns]] + ExpZ_b_R[:, [nr]]
                print("ExpZ_b[", ns, "]:\n", ExpZ_b[:, [ns]])
                ExpZZT_b[[ns], :, :] = (K + ExpZ_b[:, [ns]].dot(ExpZ_b[:, [ns]].T))[np.newaxis, :, :]
                print("ExpZZT_b[", ns, "]:\n", ExpZZT_b[[ns], :, :])
                # ExpZZT_b[[ns], :, :] = (K + ExpZ_b_S[:, [ns]].dot(ExpZ_b_S[:, [ns]].T) + \
                #                         ExpZ_b_R[:, [nr]].dot(ExpZ_b_S[:, [ns]].T) + \
                #                         ExpZ_b_S[:, [ns]].dot(ExpZ_b_R[:, [nr]].T))[np.newaxis, :, :] + \
                #                        ExpZZT_b_R[[nr], :, :]
                # UL[[ns], :, :] = XS_b_mu[:, [ns]].dot(ExpZ_b_S[:, [ns]].T)
                # UR[[ns], :, :] = XS_b_mu[:, [ns]].dot(ExpZ_b_R[:, [nr]].T)
                # U[[ns], :, :] = UL[[ns], :, :] + UR[[ns], :, :]
                U[[ns], :, :] = XS_b_mu[:, [ns]].dot(ExpZ_b[:, [ns]].T)
                # LL[[ns], :, :] = XR_b_mu[:, [nr]].dot(ExpZ_b_S[:, [ns]].T)
                # L[[ns], :, :] = LL[[ns], :, :] + LR[[nr], :, :]
                L[[ns], :, :] = XR_b_mu[:, [nr]].dot(ExpZ_b[:, [ns]].T)
                W_b_p1[[ns], :, :] = np.hstack((U[[ns], :, :], L[[ns], :, :]))
                print("W_b_p1[[", ns, "] :, :]:\n", W_b_p1[[ns], :, :])
                ns = ns + 1
            nr = nr + 1
        print(NS_b == ns)
        print("XR_b_mu:\n", XR_b_mu, XR_b_mu.shape)
        print("XS_b_mu:\n", XS_b_mu, XS_b_mu.shape)
        # ExpZ_b= P x NS_b
        print("ExpZ_b:\n", ExpZ_b, ExpZ_b.shape)
        # ExpZZ_b=NS_b x P x P
        print("ExpZZT_b:\n", ExpZZT_b, ExpZZT_b.shape)
        # W_b_p1=NS_b x D x P
        # W_b_p1_sum= D x P
        W_b_p1_sum = np.sum(W_b_p1, axis=0)
        # W_b_p2_sum= P x P
        W_b_p2_sum = np.sum(ExpZZT_b, axis=0)
        print("W_b_p1_sum:\n", W_b_p1_sum, W_b_p1_sum.shape)
        print("W_b_p2_sum:\n", W_b_p2_sum, W_b_p2_sum.shape)
        print("ns:", ns)
        return W_b_p1_sum, W_b_p2_sum, ExpZ_b, ExpZZT_b

    def calculate_Sigma2(self, W_new):
        print("calculate Sigma2")
        muS = self.muS
        muR = self.muR
        W = self.W
        print("current W :\n", W)
        sigma2 = self.sigma2
        P = self.P
        XS_b = self.XS_b
        XR_b = self.XR_b
        NS_b = self.NS_b
        NR_b = self.NR_b
        DS = self.DS
        DR = self.DR
        D = self.D
        matching_cnt = self.matching_cnt
        XR_b_mu = np.zeros((DR, NR_b))
        XS_b_mu = np.zeros((DS, NS_b))
        ExpZ_b_R = np.zeros((P, NR_b))
        ExpZ_b_S = np.zeros((P, NS_b))
        ExpZ_b = np.zeros((P, NS_b))
        ExpZZT_b = np.zeros((NS_b, P, P))

        Sigma_b_p1_R = np.zeros((1, NR_b))
        Sigma_b_p1_S = np.zeros((1, NS_b))
        Sigma_b_p1 = np.zeros((1, NS_b))
        # M = PxP
        M = W.T.dot(W) + sigma2 * np.eye(P)
        M_1 = np.linalg.inv(M)
        # G = M*D
        G = M_1.dot(W.T)
        # print("G:\n", G)
        # GS = PxDS
        GS = G[:, 0:DS]
        print("GS", GS)
        # GR = PxDR
        GR = G[:, DS:D]
        print("GR", GR)
        # K= PxP
        K = sigma2 * M_1
        print("K", K)
        ns = 0
        nr = 0
        for nr_m in range(NR_b):
            print("nr:", nr)
            # DR x 1 XR_b_mu[:, [nr]] can be used repeatedly
            XR_b_mu[:, [nr]] = (XR_b[[nr], :] - muR).T
            print("XR_b_mu[", nr, "]:\n", XR_b_mu[:, [nr]])
            # Px1  ExpZ_b_R[:, [nr]] can be used repeatedly
            ExpZ_b_R[:, [nr]] = GR.dot(XR_b_mu[:, [nr]])
            print("ExpZ_b_R[", nr, "]:\n", ExpZ_b_R[:, [nr]])
            Sigma_b_p1_R[:, [nr]] = np.sum(np.power(XR_b_mu[:, [nr]], 2), axis=0)
            print("Sigma_b_p1_R[", nr, "]:\n", Sigma_b_p1_R[:, [nr]])
            for ns_m in range(matching_cnt[nr]):
                print("ns:", ns)
                # DS x 1
                XS_b_mu[:, [ns]] = (XS_b[[ns], :] - muS).T
                ExpZ_b_S[:, [ns]] = GS.dot(XS_b_mu[:, [ns]])
                print("ExpZ_b_S[", ns, "]:\n", ExpZ_b_S[:, [ns]])
                ExpZ_b[:, [ns]] = ExpZ_b_S[:, [ns]] + ExpZ_b_R[:, [nr]]
                print("ExpZ_b[", ns, "]:\n", ExpZ_b[:, [ns]])
                ExpZZT_b[[ns], :, :] = (K + ExpZ_b[:, [ns]].dot(ExpZ_b[:, [ns]].T))[np.newaxis, :, :]
                print("ExpZZT_b[", ns, "]:\n", ExpZZT_b[[ns], :, :])
                Sigma_b_p1_S[:, [ns]] = np.sum(np.power(XS_b_mu[:, [ns]], 2), axis=0)
                # print("Sigma_b_p1_S[", ns, "]:\n", Sigma_b_p1_S[:, [ns]])
                # not change for every iteration!!!
                Sigma_b_p1[:, [ns]] = Sigma_b_p1_S[:, [ns]] + Sigma_b_p1_R[:, [nr]]
                # print("Sigma_b_p1[", ns, "]:\n", Sigma_b_p1[:, [ns]])

                ns = ns + 1
            nr = nr + 1
        print(NS_b == ns)
        print("XR_b_mu:\n", XR_b_mu, XR_b_mu.shape)
        print("XS_b_mu:\n", XS_b_mu, XS_b_mu.shape)
        # ExpZ_b= P x NS_b
        print("ExpZ_b:\n", ExpZ_b, ExpZ_b.shape)
        # ExpZZ_b=NS_b x P x P
        print("ExpZZT_b:\n", ExpZZT_b, ExpZZT_b.shape)
        Sigma_b_p1_sum = np.sum(Sigma_b_p1, axis=1)
        print("Sigma_b_p1_sum", Sigma_b_p1_sum, Sigma_b_p1_sum.shape)
        return Sigma_b_p1_sum
