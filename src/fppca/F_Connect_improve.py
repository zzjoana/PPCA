import time
from math import ceil
from src.fppca.F_improve import FPPCA

import numpy as np
import psycopg2


def fetch_count_mu():
    cursor = conn.cursor()
    sql_NR = """select * from "r7";"""
    cursor.execute(sql_NR)
    NR = cursor.rowcount
    # print("NR:", NR)
    cursor.close()

    cursor = conn.cursor()
    sql_NS = """select * from "s2";"""
    cursor.execute(sql_NS)
    NS = cursor.rowcount
    # print("NS:", NS)
    cursor.close()

    cursor = conn.cursor()
    sql_muS = """select avg("xs1"), avg("xs2") from "s1";"""
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
    cursor.close()

    cursor = conn.cursor()
    sql_muR = """select sum("r8"."xr1" * tmp.CNT), sum("r8"."xr2" * tmp.CNT),
    sum("r8"."xr3" * tmp.CNT),sum("r8"."xr4" * tmp.CNT),sum("r8"."xr5" * tmp.CNT),
    sum("r8"."xr6" * tmp.CNT),sum("r8"."xr7" * tmp.CNT),sum("r8"."xr8" * tmp.CNT),
    sum("r8"."xr9" * tmp.CNT),sum("r8"."xr10" * tmp.CNT),sum("r8"."xr11" * tmp.CNT),
    sum("r8"."xr12" * tmp.CNT),sum("r8"."xr13" * tmp.CNT),sum("r8"."xr14" * tmp.CNT),sum("r8"."xr15" * tmp.CNT)
    from "r8" join (select "s1"."fk", count(*) as CNT from "s1" GROUP BY "s1"."fk") as tmp
                      on "r8"."rid" = tmp."fk";"""
    cursor.execute(sql_muR)
    sumR = cursor.fetchone()
    # the type is tuple
    # print("sumR:\n", sumR, type(sumR))
    # change tuple to array and divide NS
    muR = np.array(sumR, dtype=float) / NS
    # print("muR:\n", muR, type(muR))
    cursor.close()
    return NR, NS, muR, muS


def fetch_data(cursorR, sql_S):
    # runtime_start_line = time.time()
    R_b_list = cursorR.fetchmany(NR_b)
    # runtime_end_1 = time.time()
    # print("runtime of 1:", (runtime_end_1 - runtime_start_line))
    R_b_array = np.array(R_b_list)
    # runtime_end_2 = time.time()
    # print("runtime of 2:", (runtime_end_2 - runtime_start_line))
    data_XR_b = np.mat(R_b_list, dtype=float)[:, 1:16]
    # runtime_end_3 = time.time()
    # print("runtime of 3:", (runtime_end_3 - runtime_start_line))
    # print("data_XR_b:\n", data_XR_b, type(data_XR_b))
    R_b_id = R_b_array[:, 0].tolist()
    # runtime_end_4 = time.time()
    # print("runtime of 4:", (runtime_end_4 - runtime_start_line))
    # R_b_id_tuple = tuple(R_b_id)
    cursorS = conn.cursor()
    # runtime_end_5 = time.time()
    # print("runtime of 5:", (runtime_end_5 - runtime_start_line))
    cursorS.execute(sql_S, [tuple(R_b_id)])
    # cursorS.execute(sql_S, [R_b_id_tuple[0], R_b_id_tuple[len(R_b_id_tuple) - 1]])
    # runtime_end_6 = time.time()
    # print("runtime of 6:", (runtime_end_6 - runtime_start_line))
    S_b_list = cursorS.fetchall()
    # runtime_end_7 = time.time()
    # print("runtime of 7:", (runtime_end_7 - runtime_start_line))
    # S_b_list_array = np.array(S_b_list)
    # runtime_end_8 = time.time()
    # print("runtime of 8:", (runtime_end_8 - runtime_start_line))
    # data_XS_b_FK_sorted = S_b_list_array[np.argsort(S_b_list_array[:, 4])]
    # runtime_end_9 = time.time()
    # print("runtime of 9:", (runtime_end_9 - runtime_start_line))
    data_XS_b_FK = np.mat(S_b_list, dtype=float)
    # runtime_end_10 = time.time()
    # print("runtime of 10:", (runtime_end_10 - runtime_start_line))
    # print("data_XS_b_FK:\n", data_XS_b_FK, type(data_XS_b_FK))
    cursorS.close()
    # runtime_end_11 = time.time()
    # print("runtime of 11:", (runtime_end_11 - runtime_start_line))
    # print("runtime of fetch data:", (runtime_end_fetch - runtime_start_fetch))
    return data_XR_b, data_XS_b_FK


def iterate_and_calculate(W, Sigma, max_iter):
    sql_R = """select "rid","xr1", "xr2", "xr3", "xr4", "xr5","xr6","xr7","xr8","xr9","xr10",
               "xr11", "xr12", "xr13", "xr14", "xr15" 
               from "r8" order by "rid";"""
    sql_S = """select  "xs1","xs2", "fk" from "s1" join (select * from "r8" where "r8"."rid" in %s ) 
               as tmp on "s1"."fk"= tmp."rid" order by "rid";"""

    for i in range(max_iter):
        print("******************************************************** iteration:" + str(i))
        # print("current W: \n", W)
        # print("current Sigma: \n", Sigma)

        fppca = FPPCA(W=W, Sigma=Sigma, P=P, DS=DS, DR=DR, muS=muS, muR=muR)

        # ##################first read data#####################
        cursorR = conn.cursor()
        cursorR.execute(sql_R)
        W_p1_sum = np.zeros((D, P))
        W_p2_sum = np.zeros((P, P))
        batch_num = 1
        while batch_num <= total_batch:
            # runtime_start_batch = time.time()
            data_XR_b, data_XS_b_FK = fetch_data(cursorR, sql_S)
            # runtime_end_fetch = time.time()
            # print("runtime of fetch:", (runtime_end_fetch - runtime_start_batch))
            # runtime_start_calculate = time.time()
            fppca.fit(data_XS_b_FK, data_XR_b)
            # print("########################################calculate E and W for batch_num:", batch_num)
            W_b_p1_sum, W_b_p2_sum = fppca.calculate_E_W()
            W_p1_sum = W_p1_sum + W_b_p1_sum
            W_p2_sum = W_p2_sum + W_b_p2_sum
            batch_num += 1
            # runtime_end_batch = time.time()
            # print("runtime of this calculate:", (runtime_end_batch - runtime_start_calculate))
            # print("runtime of this batch:", (runtime_end_batch - runtime_start_batch))
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
        Wnew = W_p1_sum.dot(np.linalg.inv(W_p2_sum))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Wnew:\n", Wnew, Wnew.shape)
        cursorR.close()

        # ##################Second read data#####################
        cursorR = conn.cursor()
        cursorR.execute(sql_R)
        Sigma_p1_sum = 0
        Sigma_p2_sum = 0
        Sigma_p3_sum = 0
        batch_num = 1
        while batch_num <= total_batch:
            data_XR_b, data_XS_b_FK = fetch_data(cursorR, sql_S)
            fppca.fit(data_XS_b_FK, data_XR_b)
            print("########################################calculate Sigma for batch_num:", batch_num)
            Sigma_b_p1_sum, Sigma_b_p2_sum, Sigma_b_p3_sum = fppca.calculate_Sigma2(Wnew)
            Sigma_p1_sum = Sigma_p1_sum + Sigma_b_p1_sum
            Sigma_p2_sum = Sigma_p2_sum + Sigma_b_p2_sum
            Sigma_p3_sum = Sigma_p3_sum + Sigma_b_p3_sum
            batch_num += 1
        # print("Sigma_p1_sum:", Sigma_p1_sum)
        # print("Sigma_p2_sum:", Sigma_p2_sum)
        # print("Sigma_p3_sum:", Sigma_p3_sum)
        Sigmanew = (1 / (NS * D)) * (Sigma_p1_sum - Sigma_p2_sum + Sigma_p3_sum)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Sigmanew:\n", Sigmanew)
        cursorR.close()
        W = Wnew
        Sigma = Sigmanew


runtime_start = time.time()
cpu_start = time.process_time()

conn = psycopg2.connect(database="syntheticdata", user="postgres", password="123456", host="localhost", port="5432")
NR, NS, muR, muS = fetch_count_mu()
# print("NR:", NR)
# print("NS:", NS)
# print("muR:\n", muR)
# print("muS:\n", muS, muS.dtype)

NR_b = 25
total_batch = ceil(NR / NR_b)
# print("total_batch:", total_batch)
P = 3
D = 17
DS = 2
DR = 15
np.random.seed(2)
W = np.random.rand(D, P)
# print('W_init:\n', W)
Sigma = 1
max_iter = 10
iterate_and_calculate(W=W, Sigma=Sigma, max_iter=max_iter)
conn.close()

runtime_end = time.time()
cpu_end = time.process_time()
print("runtime of FPPCA—improve:", (runtime_end - runtime_start))
print("cpu of FPPCA-improve:", (cpu_end - cpu_start))
# CPU profiling
