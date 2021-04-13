import time
from math import ceil

import numpy as np
import psycopg2

from src.fppca.S_improve import SPPCA


def fetch_count_mu(sql_NR, sql_N, sql_mu):
    cursor = conn.cursor()
    cursor.execute(sql_NR)
    NR = cursor.rowcount
    # print("NR:", NR)
    cursor.close()

    cursor = conn.cursor()
    cursor.execute(sql_N)
    N = cursor.rowcount
    cursor.close()

    cursor = conn.cursor()
    cursor.execute(sql_mu)
    mu_tuple = cursor.fetchone()
    mu = np.array(mu_tuple, dtype=float)
    cursor.close()
    return N, NR, mu


def fetch_data(cursorR, sql_b, NR_b):
    R_b_id = cursorR.fetchmany(NR_b)
    # R_b_array = np.array(R_b_list)
    # print("R_b_array",R_b_array)
    # R_b_id = R_b_array[:, 0].tolist()
    # print("R_b_id", len(R_b_id))
    cursorS = conn.cursor()
    start = time.time()
    cursorS.execute(sql_b, [tuple(R_b_id)])
    end = time.time()
    # print("time:", end-start)
    data_b_list = cursorS.fetchall()
    data_X_b_RID = np.mat(data_b_list, dtype=float)

    # print("data_X_b:\n", data_X_b, type(data_X_b))
    cursorS.close()
    return data_X_b_RID


def iterate_and_calculate(sql_R, sql_b, D, NR_b, W, Sigma, max_iter):
    # sql_b = """select  "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
    # "xr11", "xr12", "xr13", "xr14", "xr15"
    # from "s2" join (select * from "r7" where "r7"."rid" in %s ) as tmp on "s2"."fk"= tmp."rid" order by "rid";"""
    fetch_sum = 0
    calculate_sum = 0
    for i in range(max_iter):
        print("******************************************************** iteration:" + str(i))
        # print("current W: \n", W)
        # print("current Sigma: \n", Sigma)
        sppca = SPPCA(P=P, W=W, Sigma=Sigma, D=D, mu=mu)
        # ##################first read data#####################
        start_execute_time_1 = time.time()
        cursorR = conn.cursor()
        cursorR.execute(sql_R)
        end_execute_time_1 = time.time()
        exceute_time_1 = end_execute_time_1 - start_execute_time_1
        W_p1_sum = np.zeros((D, P))
        W_p2_sum = np.zeros((P, P))
        batch_num = 1
        fetch_for_first = 0
        calculate_for_first = 0
        while batch_num <= total_batch:
            start_fetch_1 = time.time()
            data_X_b_RID = fetch_data(cursorR, sql_b, NR_b)
            # print("data_X_b_RID:",data_X_b_RID,data_X_b_RID.shape)
            end_fetch_1 = time.time()
            fetch_time_batch_1 = end_fetch_1 - start_fetch_1
            start_calculate_1 = time.time()
            sppca.fit(data_X_b_RID)
            W_b_p1_sum, W_b_p2_sum = sppca.calculate_E_W()
            W_p1_sum = W_p1_sum + W_b_p1_sum
            W_p2_sum = W_p2_sum + W_b_p2_sum
            end_calculate_1 = time.time()
            calculate_time_batch_1 = end_calculate_1 - start_calculate_1

            fetch_for_first = fetch_for_first + fetch_time_batch_1
            calculate_for_first = calculate_for_first + calculate_time_batch_1
            batch_num += 1

        Wnew = W_p1_sum.dot(np.linalg.inv(W_p2_sum))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Wnew:\n", Wnew, Wnew.shape)
        cursorR.close()
        # ##################Second read data#####################
        start_execute_time_2 = time.time()
        cursorR = conn.cursor()
        cursorR.execute(sql_R)
        end_execute_time_2 = time.time()
        execute_time_2 = end_execute_time_2 - start_execute_time_2
        Sigma_p1_sum = 0
        Sigma_p2_sum = 0
        Sigma_p3_sum = 0
        batch_num = 1
        fetch_for_second = 0
        calculate_for_second = 0
        while batch_num <= total_batch:
            start_fetch_2 = time.time()
            data_X_b_RID = fetch_data(cursorR, sql_b, NR_b)
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

            fetch_for_second = fetch_for_second + fetch_time_batch_2
            calculate_for_second = calculate_for_second + calculate_time_batch_2
            batch_num += 1
        # print("Sigma_p1_sum:", Sigma_p1_sum)
        # print("Sigma_p2_sum:", Sigma_p2_sum)
        # print("Sigma_p3_sum:", Sigma_p3_sum)
        Sigmanew = (1 / (N * D)) * (Sigma_p1_sum - Sigma_p2_sum + Sigma_p3_sum)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Sigmanew:\n", Sigmanew)
        cursorR.close()
        W = Wnew
        Sigma = Sigmanew
        fetch_sum = fetch_sum + fetch_for_first + fetch_for_second + exceute_time_1 + execute_time_2
        calculate_sum = calculate_sum + calculate_for_first + calculate_for_second
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!final Sigmanew:\n", Sigmanew)
    return fetch_sum, calculate_sum

# expedia1_r_hotels
# expedia1_s_listings

NR_1 = """select * from "expedia1_r_hotels";"""
N_1 = """select * from "expedia1_s_listings";"""
mu_1 = """select  avg("xs1"), avg("xs2"), avg("xs3"), avg("xs4"), avg("xs5"), avg("xs6"), avg("xs7"),
         avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), avg("xr7"), avg("xr8")
         from (select * from "expedia1_s_listings" join "expedia1_r_hotels" on
          "expedia1_s_listings"."fk"= "expedia1_r_hotels"."rid")  as tmp;"""
R_1 = """select "rid" from "expedia1_r_hotels" ;"""
b_1 = """select "xs1", "xs2","xs3", "xs4","xs5", "xs6", "xs7","rid", 
          "xr1", "xr2", "xr3", "xr4", "xr5", "xr6", "xr7", "xr8"
        from "expedia1_s_listings" as tmp1,"expedia1_r_hotels" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

# expedia2_r_searches
# expedia2_s_listings

NR_2 = """select * from "expedia2_r_searches";"""
N_2 = """select * from "expedia2_s_listings";"""
mu_2 = """select  avg("xs1"), avg("xs2"), avg("xs3"), avg("xs4"), avg("xs5"), avg("xs6"), avg("xs7"),
         avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), avg("xr7"), avg("xr8"), 
         avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"), avg("xr14")
         from (select * from "expedia2_s_listings" join "expedia2_r_searches" on 
          "expedia2_s_listings"."fk"= "expedia2_r_searches"."rid") as tmp;"""
R_2 = """select "rid" from "expedia2_r_searches" ;"""
b_2 = """select "xs1", "xs2","xs3", "xs4","xs5", "xs6", "xs7","rid", 
          "xr1", "xr2", "xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10", "xr11", "xr12", "xr13", "xr14"
        from "expedia2_s_listings" as tmp1,"expedia2_r_searches" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

# movie_r_searches
# movie_s_ratings

NR_3 = """select * from "movie_r_searches";"""
N_3 = """select * from "movie_s_ratings";"""
mu_3 = """select  avg("xs1"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),avg("xr6"), avg("xr7"),
           avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"), avg("xr14"), avg("xr15"),
           avg("xr16"),avg("xr17"),avg("xr18"), avg("xr19"), avg("xr20"), avg("xr21")
           from (select * from "movie_s_ratings" join "movie_r_searches" on 
           "movie_s_ratings"."fk"= "movie_r_searches"."rid") as tmp;"""
R_3 = """select "rid" from "movie_r_searches" ;"""
b_3 = """select "xs1", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
          "xr11", "xr12", "xr13", "xr14", "xr15" , "xr16", "xr17", "xr18", "xr19", "xr20", "xr21"
        from "movie_s_ratings" as tmp1,"movie_r_searches" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

# walmart_r_indicators
# walmart_s_sales

NR_4 = """select * from "walmart_r_indicators";"""
N_4 = """select * from "walmart_s_sales";"""
mu_4 = """select  avg("xs1"), avg("xs2"),avg("xs3"),avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9")
           from (select * from "walmart_s_sales" join "walmart_r_indicators" on 
          "walmart_s_sales"."fk"= "walmart_r_indicators"."rid") as tmp;"""
R_4 = """select "rid" from "walmart_r_indicators" ;"""
b_4 = """select "xs1", "xs2","xs3", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9"
        from "walmart_s_sales" as tmp1,"walmart_r_indicators" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""


NR_sql_list = [NR_1, NR_2, NR_3, NR_4]
N_sql_list = [N_1, N_2, N_3, N_4]
mu_sql_list = [mu_1, mu_2, mu_3, mu_4]
R_sql_list = [R_1, R_2, R_3, R_4]
b_sql_list = [b_1, b_2, b_3, b_4]

D_list = [15, 21, 22, 12]
runtime_list = []
cputime_list = []
fetchdatatime_list = []
calculatetime_list = []
NR_b_list = [25, 25, 25, 25]

for i in range(2,len(NR_sql_list)):
    print("i=", i)
    runtime_start = time.time()
    # cpu_start = time.process_time()
    preparation_start = time.time()

    conn = psycopg2.connect(database="experiment_real", user="postgres", password="123456", host="localhost",
                            port="5432")
    N, NR, mu = fetch_count_mu(NR_sql_list[i], N_sql_list[i], mu_sql_list[i])
    P = 3
    total_batch = ceil(NR / NR_b_list[i])
    preparation_end = time.time()
    preparation_time = preparation_end - preparation_start
    print("Prepare for SPPCA:", (preparation_end - preparation_start))
    np.random.seed(2)
    W = np.random.rand(D_list[i], P)
    Sigma = 1
    max_iter = 10
    fetch_sum, calculate_sum = iterate_and_calculate(sql_R=R_sql_list[i], sql_b=b_sql_list[i], D=D_list[i],
                                                     NR_b=NR_b_list[i], W=W, Sigma=Sigma, max_iter=max_iter)
    conn.close()

    runtime_end = time.time()
    # cpu_end = time.process_time()
    print("runtime of SPPCA for ", i, " cases:", (runtime_end - runtime_start))
    # print("cpu of SPPCA for ", i, " cases:", (cpu_end - cpu_start))
    print("fetch data time of SPPCA for ", i, " cases:", fetch_sum + preparation_time)
    print("calculate time of SPPCA for ", i, " cases:", calculate_sum)
    runtime_list.append(runtime_end - runtime_start)
    # cputime_list.append(cpu_end - cpu_start)
    fetchdatatime_list.append(fetch_sum + preparation_time)
    calculatetime_list.append(calculate_sum)

print("runtime_list:\n", runtime_list)
# print("cputime_list:\n", cputime_list)
print("fetchdatatime_list:\n", fetchdatatime_list)
print("calculatetime_list:\n", calculatetime_list)
