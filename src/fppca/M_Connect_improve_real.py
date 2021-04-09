import decimal
import time
from math import ceil

import numpy as np
import psycopg2
from src.fppca.S_improve import SPPCA
from src.fppca.M_improve import MPPCA


def join_tables(sql_join):
    cursor = conn.cursor()
    cursor.execute(sql_join)
    cursor.close()
    conn.commit()


def drop_table():
    cursor = conn.cursor()
    sql_drop = """drop table join_result;"""
    cursor.execute(sql_drop)
    conn.commit()


def fetch_count_mu(sql_NR, sql_mu):
    cursor = conn.cursor()
    sql_T = """select * from "join_result";"""
    cursor.execute(sql_T)
    N = cursor.rowcount
    cursor.close()

    cursor = conn.cursor()
    cursor.execute(sql_NR)
    NR = cursor.rowcount
    # print("NR:", NR)
    cursor.close()

    cursor = conn.cursor()
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


def fetch_data(cursor, b, NR_b):
    # print("b", b)
    b_list = cursor.fetchmany(sum(count[b:b + NR_b, :])[0])
    # print ("sum(count[",b,":",b + NR_b," :])[0]", sum(count[b:b + NR_b, :])[0])
    data_X_b_RID = np.mat(b_list, dtype=float)
    # print("data_X_b_RID:\n", data_X_b_RID, type(data_X_b_RID))
    return data_X_b_RID


def iterate_and_calculate(sql_data, D, NR_b, W, Sigma, max_iter):
    fetch_sum = 0
    calculate_sum = 0

    for i in range(max_iter):
        print("******************************************************** iteration:" + str(i))
        sppca = SPPCA(P=P, W=W, Sigma=Sigma, D=D, mu=mu)
        # ##################first read data#####################
        start_execute_time_1 = time.time()
        cursor = conn.cursor()
        cursor.execute(sql_data)
        end_execute_time_1 = time.time()
        exceute_time_1 = end_execute_time_1 - start_execute_time_1
        W_p1_sum = np.zeros((D, P))
        W_p2_sum = np.zeros((P, P))
        batch_num = 1
        b = 0
        fetch_for_1 = 0
        calculate_for_1 = 0
        while batch_num <= total_batch:
            start_fetch_1 = time.time()
            data_X_b_RID = fetch_data(cursor, b, NR_b)
            # print("data_X_b_RID:", data_X_b_RID, data_X_b_RID.shape)
            b = b + NR_b
            end_fetch_1 = time.time()
            fetch_time_batch_1 = end_fetch_1 - start_fetch_1
            start_calculate_1 = time.time()
            sppca.fit(data_X_b_RID)
            W_b_p1_sum, W_b_p2_sum = sppca.calculate_E_W()
            W_p1_sum = W_p1_sum + W_b_p1_sum
            W_p2_sum = W_p2_sum + W_b_p2_sum
            end_calculate_1 = time.time()
            calculate_time_batch_1 = end_calculate_1 - start_calculate_1
            # print("calculate_time_batch_1:",calculate_time_batch_1)

            fetch_for_1 = fetch_for_1 + fetch_time_batch_1
            calculate_for_1 = calculate_for_1 + calculate_time_batch_1
            batch_num += 1
        Wnew = W_p1_sum.dot(np.linalg.inv(W_p2_sum))
        # print("calculate_for_1:", calculate_for_1)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Wnew:\n", Wnew, Wnew.shape)
        cursor.close()
        # ##################Second read data#####################
        start_execute_time_2 = time.time()
        cursor = conn.cursor()
        cursor.execute(sql_data)
        end_execute_time_2 = time.time()
        execute_time_2 = end_execute_time_2 - start_execute_time_2
        Sigma_p1_sum = 0
        Sigma_p2_sum = 0
        Sigma_p3_sum = 0
        batch_num = 1
        b = 0
        fetch_for_2 = 0
        calculate_for_2 = 0
        while batch_num <= total_batch:
            start_fetch_2 = time.time()
            data_X_b_RID = fetch_data(cursor, b, NR_b)
            b = b + NR_b
            end_fetch_2 = time.time()
            fetch_time_batch_2 = end_fetch_2 - start_fetch_2
            start_calculate_2 = time.time()

            sppca.fit(data_X_b_RID)
            Sigma_b_p1_sum, Sigma_b_p2_sum, Sigma_b_p3_sum = sppca.calculate_Sigma2(Wnew)
            Sigma_p1_sum = Sigma_p1_sum + Sigma_b_p1_sum
            Sigma_p2_sum = Sigma_p2_sum + Sigma_b_p2_sum
            Sigma_p3_sum = Sigma_p3_sum + Sigma_b_p3_sum
            end_calculate_2 = time.time()
            calculate_time_batch_2 = end_calculate_2 - start_calculate_2
            # print("calculate_time_batch_2:", calculate_time_batch_2)

            fetch_for_2 = fetch_for_2 + fetch_time_batch_2
            calculate_for_2 = calculate_for_2 + calculate_time_batch_2
            batch_num += 1
        # print("calculate_for_2:", calculate_for_2)
        Sigmanew = (1 / (N * D)) * (Sigma_p1_sum - Sigma_p2_sum + Sigma_p3_sum)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Sigmanew:\n", Sigmanew)
        cursor.close()
        W = Wnew
        Sigma = Sigmanew
        fetch_sum = fetch_sum + fetch_for_1 + fetch_for_2 + exceute_time_1 + execute_time_2
        calculate_sum = calculate_sum + calculate_for_1 + calculate_for_2

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!final Sigmanew:\n", Sigmanew)
    return fetch_sum, calculate_sum


# sql_join = """create table join_result AS select * from "s2" join "r7" on "s2"."fk"= "r7"."rid" order by "rid";"""
# sql_NR = """select * from "r7";"""
# sql_mu = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),
#              avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"),
#              avg("xr11"), avg("xr12"), avg("xr13"), avg("xr14"), avg("xr15")
#              from "join_result";"""
# sql_data = """select  "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
#          "xr11", "xr12", "xr13", "xr14", "xr15"
#           from "join_result";"""
# D = 17
# NR_b = 25

# expedia1_r_hotels
# expedia1_s_listings

join_1 = """create table join_result AS select * from "expedia1_s_listings" join "expedia1_r_hotels" on 
            "expedia1_s_listings"."fk"= "expedia1_r_hotels"."rid" order by "rid";"""
NR_1 = """select * from "expedia1_r_hotels";"""
mu_1 = """select avg("xs1"), avg("xs2"),avg("xs3"),avg("xs4"),avg("xs5"),avg("xs6"),avg("xs7"),
           avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"),avg("xr5"), avg("xr6"),avg("xr7"), avg("xr8")
          from "join_result";"""
data_1 = """select  "xs1", "xs2","xs3", "xs4","xs5", "xs6", "xs7","rid", 
          "xr1", "xr2", "xr3", "xr4", "xr5", "xr6", "xr7", "xr8" from "join_result";"""

# expedia2_r_searches
# expedia2_s_listings

join_2 = """create table join_result AS select * from "expedia2_s_listings" join "expedia2_r_searches" 
           on "expedia2_s_listings"."fk"= "expedia2_r_searches"."rid" order by "rid";"""
NR_2 = """select * from "expedia2_r_searches";"""
mu_2 = """select avg("xs1"), avg("xs2"),avg("xs3"),avg("xs4"),avg("xs5"),avg("xs6"),avg("xs7"),
           avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), avg("xr7"), avg("xr8"),avg("xr9"),
          avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"), avg("xr14")
          from "join_result";"""
data_2 = """select  "xs1", "xs2","xs3", "xs4","xs5", "xs6", "xs7", "rid", 
           "xr1", "xr2","xr3", "xr4", "xr5", "xr6","xr7", "xr8","xr9", "xr10", "xr11", "xr12","xr13", "xr14" 
           from "join_result";"""

# movie_r_searches
# movie_s_ratings

join_3 = """create table join_result AS select * from "movie_s_ratings" join "movie_r_searches" 
            on "movie_s_ratings"."fk"= "movie_r_searches"."rid" order by "rid";"""
NR_3 = """select * from "movie_r_searches";"""
mu_3 = """select avg("xs1"), avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"),avg("xr13"), avg("xr14"), 
           avg("xr15"), avg("xr16"),  avg("xr17"), avg("xr18"), avg("xr19"), avg("xr20"), avg("xr21")
           from "join_result";"""
data_3 = """select  "xs1", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
            "xr11", "xr12","xr13", "xr14", "xr15", "xr16", "xr17", "xr18", "xr19", "xr20", "xr21"
          from "join_result";"""
# walmart_r_indicators
# walmart_s_sales

join_4 = """create table join_result AS select * from "walmart_s_sales" join "walmart_r_indicators" 
          on "walmart_s_sales"."fk"= "walmart_r_indicators"."rid" order by "rid";"""
NR_4 = """select * from "walmart_r_indicators";"""
mu_4 = """select avg("xs1"), avg("xs2"),avg("xs3"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), 
          avg("xr6"),  avg("xr7"), avg("xr8"), avg("xr9")
           from "join_result";"""
data_4 = """select  "xs1", "xs2","xs3", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9"
          from "join_result";"""

join_sql_list = [join_1, join_2, join_3, join_4]
NR_sql_list = [NR_1, NR_2, NR_3, NR_4]
mu_sql_list = [mu_1, mu_2, mu_3, mu_4]
data_sql_list = [data_1, data_2, data_3, data_4]
D_list = [15, 21, 22, 12]
runtime_list = []
cputime_list = []
fetchdatatime_list = []
calculatetime_list = []
NR_b_list = [25, 25, 25, 25]

# runtime_start_1 = runtime_start_2 = runtime_start_3 = runtime_start_4 = runtime_start_5 = 0
# cpu_start_1 = cpu_start_2 = cpu_start_3 = cpu_start_4 = cpu_start_5 = 0
# runtime_start_list = [runtime_start_1, runtime_start_2, runtime_start_3, runtime_start_4, runtime_start_5]
# cpu_start_list = [cpu_start_1, cpu_start_2, cpu_start_3, cpu_start_4, cpu_start_5]
for i in range(2,len(join_sql_list)):
    print("i=", i)
    runtime_start = time.time()
    # cpu_start = time.process_time()
    preparation_start = time.time()
    conn = psycopg2.connect(database="experiment_real", user="postgres", password="123456", host="localhost",
                            port="5432")
    if i > 0:
        drop_table()
    # join_start=time.time()
    join_tables(join_sql_list[i])
    join_end = time.time()
    # join_time=join_end-join_start
    N, NR, mu, count = fetch_count_mu(NR_sql_list[i], mu_sql_list[i])
    P = 3
    total_batch = ceil(NR / NR_b_list[i])
    preparation_end = time.time()
    preparation_time = preparation_end - preparation_start
    print("Prepare for MPPCA:", preparation_time)
    # print("join for MPPCA:", join_time)
    np.random.seed(2)
    W = np.random.rand(D_list[i], P)
    Sigma = 1
    max_iter = 10
    fetch_sum, calculate_sum = iterate_and_calculate(sql_data=data_sql_list[i], D=D_list[i], NR_b=NR_b_list[i],
                                                     W=W, Sigma=Sigma, max_iter=max_iter)
    conn.close()
    runtime_end = time.time()
    # cpu_end = time.process_time()
    print("runtime of MPPCA for ", i, " cases:", (runtime_end - runtime_start))
    # print("cpu of MPPCA for ", i, " cases:", (cpu_end - cpu_start))
    print("fetch data time of MPPCA for ", i, " cases:", fetch_sum + preparation_time)
    print("calculate time of MPPCA for ", i, " cases:", calculate_sum)
    runtime_list.append(runtime_end - runtime_start)
    # cputime_list.append(cpu_end - cpu_start)
    fetchdatatime_list.append(fetch_sum + preparation_time)
    calculatetime_list.append(calculate_sum)
print("runtime_list:\n", runtime_list)
print("fetchdatatime_list:\n", fetchdatatime_list)
print("calculatetime_list:\n", calculatetime_list)
