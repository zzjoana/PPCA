import decimal
import time
from math import ceil

import numpy as np
import psycopg2

from src.fppca.M_improve import MPPCA


def join_tables():
    cursor = conn.cursor()
    sql_join = """create table join_result 
    AS select * from "s2" join "r7" on "s2"."fk"= "r7"."rid" order by "rid";"""
    cursor.execute(sql_join)
    cursor.close()
    conn.commit()


def drop_table():
    cursor = conn.cursor()
    sql_drop = """drop table join_result;"""
    cursor.execute(sql_drop)
    conn.commit()


def fetch_count_mu():
    cursor = conn.cursor()
    sql_T = """select * from "join_result";"""
    cursor.execute(sql_T)
    N = cursor.rowcount
    cursor.close()

    cursor = conn.cursor()
    sql_NR = """select * from "r7";"""
    cursor.execute(sql_NR)
    NR = cursor.rowcount
    # print("NR:", NR)
    cursor.close()

    cursor = conn.cursor()
    sql_mu = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"), 
                avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"),
                avg("xr11"), avg("xr12"), avg("xr13"), avg("xr14"), avg("xr15")
                from "join_result";"""
    cursor.execute(sql_mu)
    mu_tuple = cursor.fetchone()
    mu = np.array(mu_tuple, dtype=float)
    cursor.close()

    cursor = conn.cursor()
    sql_C = """select count(*) from "join_result" group by "rid" order by "rid";"""
    cursor.execute(sql_C)
    count_list = cursor.fetchall()
    count = np.array(count_list)
    # print("count", count[0:2, :], sum(count[0:2, :]), sum(count[0:2, :])[0])

    cursor.close()
    return N, NR, mu, count


def fetch_data(cursor, b):
    # print("b", b)
    b_list = cursor.fetchmany(sum(count[b:b + NR_b, :])[0])
    # print ("sum(count[",b,":",b + NR_b," :])[0]", sum(count[b:b + NR_b, :])[0])
    data_X_b_RID = np.mat(b_list, dtype=float)
    # print("data_X_b_RID:\n", data_X_b_RID, type(data_X_b_RID))
    return data_X_b_RID


def iterate_and_calculate(W, Sigma, max_iter):
    sql = """select  "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
    "xr11", "xr12", "xr13", "xr14", "xr15"
             from "join_result";"""
    for i in range(max_iter):
        print("******************************************************** iteration:" + str(i))
        # print("current W: \n", W)
        # print("current Sigma: \n", Sigma)
        mppca = MPPCA(P=P, W=W, Sigma=Sigma, D=D, mu=mu)
        # ##################first read data#####################
        cursor = conn.cursor()
        cursor.execute(sql)
        W_p1_sum = np.zeros((D, P))
        W_p2_sum = np.zeros((P, P))
        batch_num = 1
        b = 0
        while batch_num <= total_batch:
            # print("########################################calculate E and W for batch_num:", batch_num)
            runtime_start_batch = time.time()
            data_X_b_RID = fetch_data(cursor, b)
            b = b + NR_b
            runtime_end_fetch = time.time()
            print("runtime of fetch:", (runtime_end_fetch - runtime_start_batch))
            mppca.fit(data_X_b_RID)
            W_b_p1_sum, W_b_p2_sum = mppca.calculate_E_W()
            W_p1_sum = W_p1_sum + W_b_p1_sum
            W_p2_sum = W_p2_sum + W_b_p2_sum
            batch_num += 1
        Wnew = W_p1_sum.dot(np.linalg.inv(W_p2_sum))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Wnew:\n", Wnew, Wnew.shape)
        cursor.close()
        # ##################Second read data#####################
        cursor = conn.cursor()
        cursor.execute(sql)
        Sigma_p1_sum = 0
        Sigma_p2_sum = 0
        Sigma_p3_sum = 0
        batch_num = 1
        b = 0
        while batch_num <= total_batch:
            # print("########################################calculate Sigma for batch_num:", batch_num)
            data_X_b_RID = fetch_data(cursor, b)
            b = b + NR_b
            mppca.fit(data_X_b_RID)
            Sigma_b_p1_sum, Sigma_b_p2_sum, Sigma_b_p3_sum = mppca.calculate_Sigma2(Wnew)
            Sigma_p1_sum = Sigma_p1_sum + Sigma_b_p1_sum
            Sigma_p2_sum = Sigma_p2_sum + Sigma_b_p2_sum
            Sigma_p3_sum = Sigma_p3_sum + Sigma_b_p3_sum
            batch_num += 1
        # print("Sigma_p1_sum:", Sigma_p1_sum)
        # print("Sigma_p2_sum:", Sigma_p2_sum)
        # print("Sigma_p3_sum:", Sigma_p3_sum)
        Sigmanew = (1 / (N * D)) * (Sigma_p1_sum - Sigma_p2_sum + Sigma_p3_sum)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Sigmanew:\n", Sigmanew)
        cursor.close()
        W = Wnew
        Sigma = Sigmanew


runtime_start = time.time()
cpu_start = time.process_time()

conn = psycopg2.connect(database="syntheticdata", user="postgres", password="123456", host="localhost", port="5432")
drop_table()
join_tables()
N, NR, mu, count = fetch_count_mu()
# print("N:", N)
# print("mu:", mu)
#print("count", count)

NR_b = 25
total_batch = ceil(NR / NR_b)
# print("total_batch:", total_batch)
P = 3
D = 17
np.random.seed(2)
W = np.random.rand(D, P)
Sigma = 1
max_iter = 10
iterate_and_calculate(W=W, Sigma=Sigma, max_iter=max_iter)
conn.close()

runtime_end = time.time()
cpu_end = time.process_time()
print("runtime of MPPCA_improve:", (runtime_end - runtime_start))
print("cpu of MPPCA_improve:", (cpu_end - cpu_start))
