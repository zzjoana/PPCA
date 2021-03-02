import time
from math import ceil

import numpy as np
import psycopg2

from src.fppca.S_improve import SPPCA


def fetch_count_mu():
    cursor = conn.cursor()
    sql_NR = """select * from "r7";"""
    cursor.execute(sql_NR)
    NR = cursor.rowcount
    # print("NR:", NR)
    cursor.close()

    cursor = conn.cursor()
    sql_N = """select * from "s2";"""
    cursor.execute(sql_N)
    N = cursor.rowcount
    cursor.close()

    cursor = conn.cursor()
    sql_mu = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), 
                avg("xr3"), avg("xr4"), avg("xr5") , avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), 
                avg("xr11"), avg("xr12"),  avg("xr13"), avg("xr14"), avg("xr15")from
                (select * from "s2" join "r7" on "s2"."fk"= "r7"."rid") as tmp;"""
    cursor.execute(sql_mu)
    mu_tuple = cursor.fetchone()
    mu = np.array(mu_tuple, dtype=float)
    cursor.close()
    return N, NR, mu


def fetch_data(cursorR, sql_b):
    R_b_list = cursorR.fetchmany(NR_b)
    R_b_array = np.array(R_b_list)
    # print("R_b_array",R_b_array)
    R_b_id = R_b_array[:, 0].tolist()
    # print("R_b_id", len(R_b_id))

    cursorS = conn.cursor()
    runtime_start_line = time.time()
    cursorS.execute(sql_b, [tuple(R_b_id)])
    runtime_end_line = time.time()
    # print("runtime of this line:", (runtime_end_line - runtime_start_line))
    S_b_list = cursorS.fetchall()
    data_X_b_RID = np.mat(S_b_list,dtype=float)

    # print("data_X_b:\n", data_X_b, type(data_X_b))
    cursorS.close()
    return data_X_b_RID


def iterate_and_calculate(W, Sigma, max_iter):
    sql_R = """select * from "r7" order by "rid";"""
    sql_b = """select  "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
    "xr11", "xr12", "xr13", "xr14", "xr15" 
    from "s2" join (select * from "r7" where "r7"."rid" in %s ) as tmp on "s2"."fk"= tmp."rid" order by "rid";"""
    for i in range(max_iter):
        print("******************************************************** iteration:" + str(i))
        # print("current W: \n", W)
        # print("current Sigma: \n", Sigma)
        sppca = SPPCA(P=P, W=W, Sigma=Sigma, D=D, mu=mu)
        # ##################first read data#####################
        cursorR = conn.cursor()
        cursorR.execute(sql_R)
        W_p1_sum = np.zeros((D, P))
        W_p2_sum = np.zeros((P, P))
        batch_num = 1
        while batch_num <= total_batch:
            runtime_start_batch = time.time()
            # print("########################################calculate E and W for batch_num:", batch_num)
            data_X_b_RID = fetch_data(cursorR, sql_b)
            runtime_end_fetch = time.time()
            # print("runtime of fetch:", (runtime_end_fetch - runtime_start_batch))
            runtime_start_calculate = time.time()
            sppca.fit(data_X_b_RID)
            W_b_p1_sum, W_b_p2_sum = sppca.calculate_E_W()
            W_p1_sum = W_p1_sum + W_b_p1_sum
            W_p2_sum = W_p2_sum + W_b_p2_sum
            batch_num += 1
            runtime_end_batch = time.time()

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
            # print("########################################calculate Sigma for batch_num:", batch_num)
            data_X_b_RID = fetch_data(cursorR, sql_b)
            sppca.fit(data_X_b_RID)
            Sigma_b_p1_sum, Sigma_b_p2_sum, Sigma_b_p3_sum = sppca.calculate_Sigma2(Wnew)
            Sigma_p1_sum = Sigma_p1_sum + Sigma_b_p1_sum
            Sigma_p2_sum = Sigma_p2_sum + Sigma_b_p2_sum
            Sigma_p3_sum = Sigma_p3_sum + Sigma_b_p3_sum
            batch_num += 1
        # print("Sigma_p1_sum:", Sigma_p1_sum)
        # print("Sigma_p2_sum:", Sigma_p2_sum)
        # print("Sigma_p3_sum:", Sigma_p3_sum)
        Sigmanew = (1 / (N * D)) * (Sigma_p1_sum - Sigma_p2_sum + Sigma_p3_sum)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Sigmanew:\n", Sigmanew)
        cursorR.close()
        W = Wnew
        Sigma = Sigmanew


runtime_start = time.time()
cpu_start = time.process_time()

conn = psycopg2.connect(database="syntheticdata", user="postgres", password="123456", host="localhost", port="5432")
N, NR, mu = fetch_count_mu()
# print("N:", N)
# print("NR:", NR)
# print("mu:", mu)
NR_b = 25
total_batch = ceil(NR / NR_b)
# print("total_batch:", total_batch)
P = 3
D = 17
np.random.seed(2)
W = np.random.rand(D, P)
# print('W_init:\n', W)
Sigma = 1
max_iter = 10
iterate_and_calculate(W=W, Sigma=Sigma, max_iter=max_iter)
conn.close()

runtime_end = time.time()
cpu_end = time.process_time()
print("runtime of SPPCA_improve:", (runtime_end - runtime_start))
print("cpu of SPPCA_improve:", (cpu_end - cpu_start))
