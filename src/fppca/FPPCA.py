from collections import Counter

import numpy as np


class FPPCA(object):
    def __init__(self, latent_dim, sigma2, max_iter, NR_b, muS, muR, batch_num):
        # P = dimensionality of the latent variable
        self.P = latent_dim
        # sigma2 = variance of the noise
        self.sigma2 = sigma2
        # NR = number of R in each batch
        self.NR_b = NR_b
        # NS = number of S in this batch
        self.NS_b = 0
        # D = dimensionality of S+R
        self.D = 0
        # DS = dimensionality of S
        self.DS = 0
        # DR = dimensionality of R
        self.DR = 0
        # muS = mean of S
        self.muS = muS
        # muR = mean of R
        self.muR = muR
        # W = projection matrix DxP
        self.W = None
        # maximum iterations to do
        self.max_iter = max_iter
        # XS is NS_b*DS
        self.XS_b = None
        # XR is NR_b*DR
        self.XR_b = None
        self.init = False
        self.matching_cnt = None
        self.batch_num = batch_num

    #    def mu_calculation(self):

    def fit(self, data_XS_b_FK, data_XR_b):
        if not self.init:
            print("FPPCA for batch_num:", self.batch_num)
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
            self.DS = data_XS_b.shape[1]
            print('DS', self.DS)
            self.DR = data_XR_b.shape[1]
            print('DR', self.DR)
            self.D = self.DS + self.DR
            print('D', self.D)
            np.random.seed(2)
            self.W = np.random.rand(self.D, self.P)
            print('W:\n', self.W)
            self.init = True
            self.expectation_maximization()

        return data_XS_b, data_XR_b

    def expectation_maximization(self):

        print("EM algorithm")
        muS = self.muS
        muR = self.muR
        W = self.W
        sigma2 = self.sigma2
        P = self.P
        XS_b = self.XS_b
        XR_b = self.XR_b
        NS_b = self.NS_b
        NR_b = self.NR_b
        DS = self.DS
        DR = self.DR
        D = self.D
        ExpZ_b_R = np.zeros((P, NR_b))
        ExpZ_b_S = np.zeros((P, NS_b))
        ExpZ_b = np.zeros((P, NS_b))
        ExpZZT_b_R = np.zeros((NR_b, P, P))
        ExpZZT_b = np.zeros((NS_b, P, P))
        matching_cnt = self.matching_cnt
        batch_num = self.batch_num

        print("E-step")
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
        print("calculate EZn and EZZn for batch_num:", batch_num)
        ns = 0
        for nr in range(NR_b):
            # Px1  ExpZ_b_R[:, [nr]] can be used repeatedly
            ExpZ_b_R[:, [nr]] = GR.dot((XR_b[[nr], :] - muR).T)
            # P x P x 1  ExpZZT_b_R[[nr], :, :] can be used repeatedly
            ExpZZT_b_R[[nr], :, :] = ExpZ_b_R[:, [nr]].dot(ExpZ_b_R[:, [nr]].T)
            print("ExpZ_b_R[", nr, "]:\n", ExpZ_b_R[:, [nr]])
            for ns_m in range(matching_cnt[nr]):
                ExpZ_b_S[:, [ns]] = GS.dot((XS_b[[ns], :] - muS).T)
                print("ExpZ_b_S[", ns, "]:\n", ExpZ_b_S[:, [ns]])
                ExpZ_b[:, [ns]] = ExpZ_b_S[:, [ns]] + ExpZ_b_R[:, [nr]]
                print("ExpZ_b[", ns, "]:\n", ExpZ_b[:, [ns]])
                # ExpZZT_b[[ns], :, :] = (K + ExpZ_b[:, [ns]].dot(ExpZ_b[:, [ns]].T))[np.newaxis, :, :]
                ExpZZT_b[[ns], :, :] = (K + ExpZ_b_S[:, [ns]].dot(ExpZ_b_S[:, [ns]].T) + \
                                        ExpZ_b_R[:, [nr]].dot(ExpZ_b_S[:, [ns]].T) + \
                                        ExpZ_b_S[:, [ns]].dot(ExpZ_b_R[:, [nr]].T))[np.newaxis, :, :] + \
                                       ExpZZT_b_R[[nr], :, :]
                print("ExpZZT_b[", ns, "]:\n", ExpZZT_b[[ns], :, :])
                ns = ns + 1
        print(NS_b == ns)
        # ExpZ_b= P x NS_b
        print("ExpZ_b_S:\n", ExpZ_b_S, ExpZ_b_S.shape)
        # ExpZ_b= P x NS_b
        print("ExpZ_b:\n", ExpZ_b, ExpZ_b.shape)
        # ExpZ_b=NS_b x P x P
        print("ExpZZT_b:\n", ExpZZT_b, ExpZZT_b.shape)
