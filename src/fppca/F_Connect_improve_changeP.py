import time
from math import ceil
from src.fppca.F_improve import FPPCA

import numpy as np
import psycopg2


def fetch_count_mu(sql_NR, sql_NS, sql_muS, sql_muR):
    cursor = conn.cursor()
    cursor.execute(sql_NR)
    NR = cursor.rowcount
    # print("NR:", NR)
    cursor.close()

    cursor = conn.cursor()
    cursor.execute(sql_NS)
    NS = cursor.rowcount
    # print("NS:", NS)
    cursor.close()

    cursor = conn.cursor()
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
    cursor.execute(sql_muR)
    sumR = cursor.fetchone()
    # the type is tuple
    # print("sumR:\n", sumR, type(sumR))
    # change tuple to array and divide NS
    muR = np.array(sumR, dtype=float) / NS
    # print("muR:\n", muR, type(muR))
    cursor.close()
    return NR, NS, muS, muR


def fetch_data(cursorR, sql_S, NR_b, DR):
    R_b_list = cursorR.fetchmany(NR_b)
    R_b_array = np.array(R_b_list)
    data_XR_b = np.mat(R_b_list, dtype=float)[:, 1:DR + 1]
    # print("data_XR_b:\n", data_XR_b, type(data_XR_b))
    R_b_id = R_b_array[:, 0].tolist()
    # R_b_id_tuple = tuple(R_b_id)
    cursorS = conn.cursor()
    cursorS.execute(sql_S, [tuple(R_b_id)])
    S_b_list = cursorS.fetchall()
    # S_b_list_array = np.array(S_b_list)
    # data_XS_b_FK_sorted = S_b_list_array[np.argsort(S_b_list_array[:, 4])]
    data_XS_b_FK = np.mat(S_b_list, dtype=float)
    cursorS.close()
    return data_XR_b, data_XS_b_FK


def iterate_and_calculate(sql_R, sql_S, D, DS, DR, NR_b, W, Sigma, max_iter):
    fetch_sum = 0
    calculate_sum = 0

    for i in range(max_iter):
        print("******************************************************** iteration:" + str(i))
        fppca = FPPCA(W=W, Sigma=Sigma, P=P, DS=DS, DR=DR, muS=muS, muR=muR)

        # ##################first read data#####################
        cursorR = conn.cursor()
        cursorR.execute(sql_R)
        W_p1_sum = np.zeros((D, P))
        W_p2_sum = np.zeros((P, P))
        batch_num = 1
        fetch_for_first = 0
        calculate_for_first = 0
        while batch_num <= total_batch:
            start_fetch_1 = time.time()
            data_XR_b, data_XS_b_FK = fetch_data(cursorR, sql_S, NR_b, DR)
            end_fetch_1 = time.time()
            fetch_time_batch_1 = end_fetch_1 - start_fetch_1
            # print("########################################calculate E and W for batch_num:", batch_num)
            start_calculate_1 = time.time()
            fppca.fit(data_XS_b_FK, data_XR_b)
            W_b_p1_sum, W_b_p2_sum = fppca.calculate_E_W()
            W_p1_sum = W_p1_sum + W_b_p1_sum
            W_p2_sum = W_p2_sum + W_b_p2_sum
            end_calculate_1 = time.time()
            calculate_time_batch_1 = end_calculate_1 - start_calculate_1
            # print("calculate_time_batch_1:", calculate_time_batch_1)

            fetch_for_first = fetch_for_first + fetch_time_batch_1
            calculate_for_first = calculate_for_first + calculate_time_batch_1
            batch_num += 1
        Wnew = W_p1_sum.dot(np.linalg.inv(W_p2_sum))
        # print("calculate_for_first:", calculate_for_first)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Wnew:\n", Wnew, Wnew.shape)
        cursorR.close()

        # ##################Second read data#####################
        cursorR = conn.cursor()
        cursorR.execute(sql_R)
        Sigma_p1_sum = 0
        Sigma_p2_sum = 0
        Sigma_p3_sum = 0
        batch_num = 1
        fetch_for_second = 0
        calculate_for_second = 0
        while batch_num <= total_batch:
            start_fetch_2 = time.time()
            data_XR_b, data_XS_b_FK = fetch_data(cursorR, sql_S, NR_b, DR)
            end_fetch_2 = time.time()
            fetch_time_batch_2 = end_fetch_2 - start_fetch_2
            # print("########################################calculate Sigma for batch_num:", batch_num)
            start_calculate_2 = time.time()
            fppca.fit(data_XS_b_FK, data_XR_b)
            Sigma_b_p1_sum, Sigma_b_p2_sum, Sigma_b_p3_sum = fppca.calculate_Sigma2(Wnew)
            Sigma_p1_sum = Sigma_p1_sum + Sigma_b_p1_sum
            Sigma_p2_sum = Sigma_p2_sum + Sigma_b_p2_sum
            Sigma_p3_sum = Sigma_p3_sum + Sigma_b_p3_sum
            end_calculate_2 = time.time()
            calculate_time_batch_2 = end_calculate_2 - start_calculate_2
            # print("calculate_time_batch_2:", calculate_time_batch_2)

            fetch_for_second = fetch_for_second + fetch_time_batch_2
            calculate_for_second = calculate_for_second + calculate_time_batch_2
            batch_num += 1
        Sigmanew = (1 / (NS * D)) * (Sigma_p1_sum - Sigma_p2_sum + Sigma_p3_sum)
        # print("calculate_for_second:", calculate_for_second)

        cursorR.close()
        W = Wnew
        Sigma = Sigmanew
        fetch_sum = fetch_sum + fetch_for_first + fetch_for_second
        calculate_sum = calculate_sum + calculate_for_first + calculate_for_second
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Sigmanew:\n", Sigmanew)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!final Sigmanew:\n", Sigmanew)
    return fetch_sum, calculate_sum





NR_1 = """select * from "r25";"""
NS_1 = """select * from "s5";"""
muS_1 = """select avg("xs1"), avg("xs2") from "s5";"""
muR_1 = """select sum("r25"."xr1" * tmp.CNT), sum("r25"."xr2" * tmp.CNT), sum("r25"."xr3" * tmp.CNT)
, sum("r25"."xr4" * tmp.CNT), sum("r25"."xr5" * tmp.CNT), sum("r25"."xr6" * tmp.CNT)
           from "r25" join (select "s5"."fk", count(*) as CNT from "s5" GROUP BY "s5"."fk") as tmp
            on "r25"."rid" = tmp."fk";"""
R_1 = """select "rid","xr1", "xr2","xr3", "xr4", "xr5"
       from "r25" order by "rid";"""
S_1 = """select  "xs1","xs2", "fk" from "s5" where "fk" in %s order by "fk"; """



NR_sql_list = [NR_1, NR_2, NR_3, NR_4, NR_5]
NS_sql_list = [NS_1, NS_2, NS_3, NS_4, NS_5]
muS_sql_list = [muS_1, muS_2, muS_3, muS_4, muS_5]
muR_sql_list = [muR_1, muR_2, muR_3, muR_4, muR_5]
R_sql_list = [R_1, R_2, R_3, R_4, R_5]
S_sql_list = [S_1, S_2, S_3, S_4, S_5]

D_list = [4, 8, 12, 16, 20]
DS_list = [2, 2, 2, 2, 2]
DR_list = [2, 6, 10, 14, 18]
NR_b_list = [25, 25, 25, 25, 25]

runtime_list = []
cputime_list = []
fetchdatatime_list = []
calculatetime_list = []
for i in range( len(NR_sql_list)):
    print("i=", i)
    runtime_start = time.time()
    # cpu_start = time.process_time()
    preparation_start = time.time()

    conn = psycopg2.connect(database="experiment_syn", user="postgres", password="123456", host="localhost",
                            port="5432")
    NR, NS, muS, muR = fetch_count_mu(NR_sql_list[i], NS_sql_list[i], muS_sql_list[i], muR_sql_list[i])

    P = 3
    total_batch = ceil(NR / NR_b_list[i])
    preparation_end = time.time()
    preparation_time = preparation_end - preparation_start
    print("Prepare for FPPCA:", (preparation_end - preparation_start))
    np.random.seed(2)
    W = np.random.rand(D_list[i], P)
    Sigma = 1
    max_iter = 10
    fetch_sum, calculate_sum = iterate_and_calculate(sql_R=R_sql_list[i], sql_S=S_sql_list[i], D=D_list[i],
                                                     DS=DS_list[i], DR=DR_list[i], NR_b=NR_b_list[i], W=W,
                                                     Sigma=Sigma, max_iter=max_iter)
    conn.close()

    runtime_end = time.time()
    # cpu_end = time.process_time()
    print("runtime of FPPCA for ", i, " cases:", (runtime_end - runtime_start))
    # print("cpu of FPPCA for ", i, " cases:", (cpu_end - cpu_start))
    print("fetch data time of FPPCA for ", i, " cases:", fetch_sum + preparation_time)
    print("calculate time of FPPCA for ", i, " cases:", calculate_sum)
    runtime_list.append(runtime_end - runtime_start)
    # cputime_list.append(cpu_end - cpu_start)
    fetchdatatime_list.append(fetch_sum + preparation_time)
    calculatetime_list.append(calculate_sum)

print("runtime_list:\n", runtime_list)
# print("cputime_list:\n", cputime_list)
print("fetchdatatime_list:\n", fetchdatatime_list)
print("calculatetime_list:\n", calculatetime_list)
