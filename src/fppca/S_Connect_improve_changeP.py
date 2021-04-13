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


def iterate_and_calculate(sql_R, sql_b,P, D, NR_b, W, Sigma, max_iter):
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



NR_1 = """select * from "r25";"""
N_1 = """select * from "s5";"""
mu_1 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6")
         from (select * from "s5" join "r25" on "s5"."fk"= "r25"."rid") as tmp;"""
R_1 = """select "rid" from "r25" ;"""
b_1 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5", "xr6"
        from "s5" as tmp1,"r25" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""


NR_sql_list = [NR_1,NR_1,NR_1,NR_1,NR_1]
N_sql_list = [N_1,N_1,N_1,N_1,N_1 ]
mu_sql_list = [mu_1,mu_1,mu_1,mu_1,mu_1 ]
R_sql_list = [R_1,R_1,R_1,R_1,R_1 ]
b_sql_list = [b_1,b_1,b_1,b_1,b_1 ]

D_list = [8,8,8,8,8]
runtime_list = []
cputime_list = []
fetchdatatime_list = []
calculatetime_list = []
NR_b_list = [25, 25, 25, 25, 25]
P_list = [2, 3, 4, 5, 6]

for i in range(len(P_list)):
    print("i=", i)
    runtime_start = time.time()
    # cpu_start = time.process_time()
    preparation_start = time.time()

    conn = psycopg2.connect(database="experiment_syn", user="postgres", password="123456", host="localhost",
                            port="5432")
    N, NR, mu = fetch_count_mu(NR_sql_list[i], N_sql_list[i], mu_sql_list[i])
    total_batch = ceil(NR / NR_b_list[i])
    preparation_end = time.time()
    preparation_time = preparation_end - preparation_start
    print("Prepare for SPPCA:", (preparation_end - preparation_start))
    np.random.seed(2)
    W = np.random.rand(D_list[i], P_list[i])
    Sigma = 1
    max_iter = 10
    fetch_sum, calculate_sum = iterate_and_calculate(sql_R=R_sql_list[i], sql_b=b_sql_list[i], P=P_list[i], D=D_list[i],
                                                     NR_b=NR_b_list[i], W=W, Sigma=Sigma, max_iter=max_iter)
    conn.close()

    runtime_end = time.time()
    # cpu_end = time.process_time()
    print("runtime of SPPCA for P=", i, " cases:", (runtime_end - runtime_start))
    # print("cpu of SPPCA for ", i, " cases:", (cpu_end - cpu_start))
    print("fetch data time of SPPCA for P=", i, " cases:", fetch_sum + preparation_time)
    print("calculate time of SPPCA for P=", i, " cases:", calculate_sum)
    runtime_list.append(runtime_end - runtime_start)
    # cputime_list.append(cpu_end - cpu_start)
    fetchdatatime_list.append(fetch_sum + preparation_time)
    calculatetime_list.append(calculate_sum)

print("runtime_list:\n", runtime_list)
# print("cputime_list:\n", cputime_list)
print("fetchdatatime_list:\n", fetchdatatime_list)
print("calculatetime_list:\n", calculatetime_list)
