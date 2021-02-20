from math import ceil
from src.fppca.FPPCA import FPPCA

import numpy as np
import psycopg2


def fetch_count_mu():
    conn = psycopg2.connect(database="F-test", user="postgres", password="123456", host="localhost", port="5432")
    cursor = conn.cursor()

    sql_NR = """select * from "R";"""
    cursor.execute(sql_NR)
    NR = cursor.rowcount
    # print("NR:", NR)

    sql_NS = """select * from "S";"""
    cursor.execute(sql_NS)
    NS = cursor.rowcount
    # print("NS:", NS)

    sql_muS = """select avg("SID"), avg("XS1"), avg("XS2") from "S";"""
    cursor.execute(sql_muS)
    # fetchone return a tuple
    muS_tuple = cursor.fetchone()
    # print("muS_tuple:\n ", muS_tuple, type(muS_tuple))
    # change tuple to array
    muS = np.array(muS_tuple, dtype=float)
    # the result of round is list then change to array again
    # muS = np.array([round(i, 2) for i in muS_array])
    # print("muS:\n", muS, muS.dtype)

    # sql_count = """select "FK", count(*) from "S" group by "FK" order by "FK"; """
    # cursor.execute(sql_count)
    # weights = cursor.fetchall()
    # print("weights:\n", weights, type(weights))

    sql_sumR = """select sum("R"."RID" * tmp.CNT),sum("R"."XR1" * tmp.CNT), sum("R"."XR2" * tmp.CNT) from "R"
                 join (select "S"."FK", count(*) as CNT from "S" GROUP BY "S"."FK") as tmp
                      on "R"."RID" = tmp."FK";"""
    cursor.execute(sql_sumR)
    sumR = cursor.fetchone()
    # the type is tuple
    # print("sumR:\n", sumR, type(sumR))
    # change tuple to array and divide NS
    muR = np.array(sumR, dtype=float) / NS
    # print("muR:\n", muR, type(muR))

    cursor.close()
    conn.close()
    return NR, NS, muR, muS


def fetch_data(cursorR, cursorS):
    R_b_list = cursorR.fetchmany(NR_b)
    R_b_array = np.array(R_b_list)
    data_XR_b = np.mat(R_b_list)
    print("data_XR_b:\n", data_XR_b, type(data_XR_b))
    R_b_id = R_b_array[:, 0].tolist()
    cursorS.execute(sql_S, [tuple(R_b_id)])
    S_b_list = cursorS.fetchall()
    data_XS_b_FK = np.mat(S_b_list)
    print("data_XS_b_FK:\n", data_XS_b_FK, type(data_XS_b_FK))
    return data_XR_b, data_XS_b_FK


def iterate_and_calculate():
    np.random.seed(2)
    W = np.random.rand(D, P)
    print('W_init:\n', W)
    sigma2 = 1
    max_iter = 2

    for i in range(max_iter):
        print("******************************************************** iteration:" + str(i))

        fppca = FPPCA(W=W, sigma2=sigma2, P=P, NR_b=NR_b, DS=DS, DR=DR, muS=muS, muR=muR)

        W_p1_sum = np.zeros((D, P))
        # print(W_p1_sum)
        W_p2_sum = np.zeros((P, P))
        # print(W_p2_sum)
        Sigma_p1_sum = 0
        ExpZ = None

        cursorR1 = conn.cursor()
        cursorS1 = conn.cursor()
        cursorR1.execute(sql_R)
        batch_num = 1
        while batch_num <= total_batch:
            data_XR_b, data_XS_b_FK = fetch_data(cursorR1, cursorS1)
            fppca.fit(data_XS_b_FK, data_XR_b)
            print("########################################calculate E and W for batch_num:", batch_num)
            W_b_p1_sum, W_b_p2_sum, ExpZ_b, ExpZZT_b = fppca.calculate_E_W()
            W_p1_sum = W_p1_sum + W_b_p1_sum
            # print("W_p1_sum:\n", W_p1_sum)
            W_p2_sum = W_p2_sum + W_b_p2_sum
            # print("W_p2_sum:\n", W_p2_sum)
            # Sigma_p1_sum = Sigma_p1_sum + Sigma_b_p1_sum
            # if ExpZ is None:
            #     ExpZ = ExpZ_b
            # else:
            #     ExpZ = np.hstack((ExpZ, ExpZ_b))
            # ExpZ = np.hstack((ExpZ, ExpZ_b))
            # print("@@@ExpZ:", ExpZ, ExpZ.shape)

            batch_num += 1

        # get new W = Dx P
        W_new = W_p1_sum.dot(np.linalg.inv(W_p2_sum))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!W_new:\n", W_new, W_new.shape)
        #print("Sigma_p1_sum:", Sigma_p1_sum)
        cursorR1.close()
        cursorS1.close()
        cursorR2 = conn.cursor()
        cursorS2 = conn.cursor()
        cursorR2.execute(sql_R)
        batch_num = 1
        while batch_num <= total_batch:
            data_XR_b, data_XS_b_FK = fetch_data(cursorR2, cursorS2)
            fppca.fit(data_XS_b_FK, data_XR_b)
            print("########################################calculate Sigma for batch_num:", batch_num)
            Sigma_b_p1_sum = fppca.calculate_Sigma2(W_new)
            Sigma_p1_sum = Sigma_p1_sum + Sigma_b_p1_sum
            batch_num += 1
        print("Sigma_p1_sum:", Sigma_p1_sum)

        cursorR2.close()
        cursorS2.close()
        W = W_new



NR, NS, muR, muS = fetch_count_mu()
print("NR:", NR)
print("NS:", NS)
print("muR:\n", muR)
print("muS:\n", muS, muS.dtype)

conn = psycopg2.connect(database="F-test", user="postgres", password="123456", host="localhost", port="5432")

sql_R = """select * from "R" order by "RID";"""
sql_S = """select * from "S" where "FK" in %s order by "FK";"""
NR_b = 2
total_batch = ceil(NR / NR_b)
print("total_batch:", total_batch)
P = 2
D = 6
DS = 3
DR = 3

iterate_and_calculate()

conn.close()
