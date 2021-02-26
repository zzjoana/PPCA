import time
from math import ceil

import numpy as np
import psycopg2

from src.fppca.SPPCA_improve import SPPCA




def fetch_count_mu():
    cursor = conn.cursor()
    sql_NR = """select * from "R";"""
    cursor.execute(sql_NR)
    NR = cursor.rowcount
    # print("NR:", NR)
    cursor.close()

    cursor = conn.cursor()
    sql_N = """select * from "S";"""
    cursor.execute(sql_N)
    N = cursor.rowcount
    cursor.close()

    cursor = conn.cursor()
    sql_mu = """select avg("SID"), avg("XS1"), avg("XS2"),avg("XS3"), avg("RID"),avg("XR1"), avg("XR2"), 
    avg("XR3"), avg("XR4"), avg("XR5"), avg("XR6"), avg("XR7"), avg("XR8") from
                (select * from "S" join "R" on "S"."FK"= "R"."RID") as tmp;"""
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
    print("runtime of this line:", (runtime_end_line - runtime_start_line))
    S_b_list = cursorS.fetchall()
    data_X_b = np.mat(S_b_list)
    # print("data_X_b:\n", data_X_b, type(data_X_b))
    cursorS.close()
    return  data_X_b



def iterate_and_calculate(W, Sigma, max_iter):
    sql_R = """select * from "R" order by "RID";"""
    sql_b = """select "SID", "XS1", "XS2","XS3", "RID", "XR1", "XR2", "XR3", "XR4", "XR5", "XR6", "XR7", "XR8"  
    from "S" join (select * from "R" where "R"."RID" in %s ) as tmp on "S"."FK"= tmp."RID" order by "RID";"""
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
            data_X_b = fetch_data(cursorR,  sql_b)
            runtime_end_fetch = time.time()
            print("runtime of fetch:", (runtime_end_fetch - runtime_start_batch))
            runtime_start_calculate= time.time()
            sppca.fit(data_X_b)
            W_b_p1_sum, W_b_p2_sum = sppca.calculate_E_W()
            W_p1_sum = W_p1_sum + W_b_p1_sum
            W_p2_sum = W_p2_sum + W_b_p2_sum
            batch_num += 1
            runtime_end_batch = time.time()

            print("runtime of this calculate:", (runtime_end_batch - runtime_start_calculate))
            print("runtime of this batch:", (runtime_end_batch - runtime_start_batch))
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
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
            print("########################################calculate Sigma for batch_num:", batch_num)
            data_X_b = fetch_data(cursorR, sql_b)
            sppca.fit(data_X_b)
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

conn = psycopg2.connect(database="realdataWalmart", user="postgres", password="123456", host="localhost", port="5432")
N,NR, mu = fetch_count_mu()
# print("N:", N)
# print("NR:", NR)
# print("mu:", mu)
NR_b = 300
total_batch = ceil(NR / NR_b)
# print("total_batch:", total_batch)
P = 2
D = 13
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