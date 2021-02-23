import time
from math import ceil

import numpy as np
import psycopg2

from src.fppca.MPPCA import MPPCA


def join_tables():
    cursor = conn.cursor()
    sql_join = """create table join_result 
    AS select * from "S" join "R" on "S"."FK"= "R"."RID";"""
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
    sql_mu = """select avg("SID"), avg("XS1"), avg("XS2"), avg("RID"),avg("XR1"), avg("XR2") from "join_result";"""
    cursor.execute(sql_mu)
    mu_tuple = cursor.fetchone()
    mu = np.array(mu_tuple, dtype=float)
    cursor.close()
    return N, mu


def fetch_data(cursor):
    b_list = cursor.fetchmany(N_b)
    data_X_b = np.mat(b_list)
    print("data_X_b:\n", data_X_b, type(data_X_b))
    return data_X_b


def iterate_and_calculate(W, Sigma, max_iter):
    sql = """select "SID", "XS1", "XS2", "RID", "XR1", "XR2" from "join_result" order by "RID";"""
    for i in range(max_iter):
        print("******************************************************** iteration:" + str(i))
        print("current W: \n", W)
        print("current Sigma: \n", Sigma)
        mppca = MPPCA(P=P, W=W, Sigma=Sigma, D=D, mu=mu)
        # ##################first read data#####################
        cursor = conn.cursor()
        cursor.execute(sql)
        W_p1_sum = np.zeros((D, P))
        W_p2_sum = np.zeros((P, P))
        batch_num = 1
        while batch_num <= total_batch:
            print("########################################calculate E and W for batch_num:", batch_num)
            data_X_b = fetch_data(cursor)
            mppca.fit(data_X_b)
            W_b_p1_sum, W_b_p2_sum = mppca.calculate_E_W()
            W_p1_sum = W_p1_sum + W_b_p1_sum
            W_p2_sum = W_p2_sum + W_b_p2_sum
            batch_num += 1
        Wnew = W_p1_sum.dot(np.linalg.inv(W_p2_sum))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Wnew:\n", Wnew, Wnew.shape)
        cursor.close()
        # ##################Second read data#####################
        cursor = conn.cursor()
        cursor.execute(sql)
        Sigma_p1_sum = 0
        Sigma_p2_sum = 0
        Sigma_p3_sum = 0
        batch_num = 1
        while batch_num <= total_batch:
            print("########################################calculate Sigma for batch_num:", batch_num)
            data_X_b = fetch_data(cursor)
            mppca.fit(data_X_b)
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

conn = psycopg2.connect(database="F-test", user="postgres", password="123456", host="localhost", port="5432")
drop_table()
join_tables()
N, mu = fetch_count_mu()
# print("N:", N)
# print("mu:", mu)
N_b = 2
total_batch = ceil(N / N_b)
# print("total_batch:", total_batch)
P = 2
D = 6
np.random.seed(2)
W = np.random.rand(D, P)
# print('W_init:\n', W)
Sigma = 1
max_iter = 100
iterate_and_calculate(W=W, Sigma=Sigma, max_iter=max_iter)
conn.close()

runtime_end = time.time()
cpu_end = time.process_time()
print("runtime of MPPCA:", (runtime_end - runtime_start))
print("cpu of MPPCA:", (cpu_end - cpu_start))
