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
join_1 = """create table join_result AS select * from "s3" join "r3" on "s3"."fk"= "r3"."rid" order by "rid";"""
NR_1 = """select * from "r3";"""
mu_1 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2")
          from "join_result";"""
data_1 = """select  "xs1", "xs2", "rid", "xr1", "xr2" from "join_result";"""

join_2 = """create table join_result AS select * from "s3" join "r11" on "s3"."fk"= "r11"."rid" order by "rid";"""
NR_2 = """select * from "r11";"""
mu_2 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6")
          from "join_result";"""
data_2 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6" from "join_result";"""

join_3 = """create table join_result AS select * from "s3" join "r8" on "s3"."fk"= "r8"."rid" order by "rid";"""
NR_3 = """select * from "r8";"""
mu_3 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10")
           from "join_result";"""
data_3 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10"
          from "join_result";"""

join_4 = """create table join_result AS select * from "s3" join "r12" on "s3"."fk"= "r12"."rid" order by "rid";"""
NR_4 = """select * from "r12";"""
mu_4 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"),avg("xr11"), avg("xr12"),avg("xr13"), avg("xr14") 
           from "join_result";"""
data_4 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
               "xr11", "xr12","xr13", "xr14"
          from "join_result";"""

join_5 = """create table join_result AS select * from "s3" join "r13" on "s3"."fk"= "r13"."rid" order by "rid";"""
NR_5 = """select * from "r13";"""
mu_5 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"),avg("xr13"), avg("xr14"),
          avg("xr15"), avg("xr16"), avg("xr17"), avg("xr18") from "join_result";"""
data_5 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
           "xr11", "xr12","xr13", "xr14", "xr15", "xr16", "xr17", "xr18"
          from "join_result";"""

join_6 = """create table join_result AS select * from "s1" join "r1" on "s1"."fk"= "r1"."rid" order by "rid";"""
NR_6 = """select * from "r1";"""
mu_6 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2")
          from "join_result";"""
data_6 = """select  "xs1", "xs2", "rid", "xr1", "xr2"
          from "join_result";"""

join_7 = """create table join_result AS select * from "s1" join "r14" on "s1"."fk"= "r14"."rid" order by "rid";"""
NR_7 = """select * from "r14";"""
mu_7 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6")
          from "join_result";"""
data_7 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6"
          from "join_result";"""

join_8 = """create table join_result AS select * from "s1" join "r6" on "s1"."fk"= "r6"."rid" order by "rid";"""
NR_8 = """select * from "r6";"""
mu_8 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10") 
           from "join_result";"""
data_8 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10"
          from "join_result";"""

join_9 = """create table join_result AS select * from "s1" join "r15" on "s1"."fk"= "r15"."rid" order by "rid";"""
NR_9 = """select * from "r15";"""
mu_9 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10") ,avg("xr11"), avg("xr12"),avg("xr13"), avg("xr14") 
           from "join_result";"""
data_9 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
           "xr11", "xr12","xr13", "xr14"
          from "join_result";"""

join_10 = """create table join_result AS select * from "s1" join "r16" on "s1"."fk"= "r16"."rid" order by "rid";"""
NR_10 = """select * from "r16";"""
mu_10 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"),avg("xr13"), avg("xr14"),
          avg("xr15"), avg("xr16"), avg("xr17"), avg("xr18") from "join_result";"""
data_10 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
           "xr11", "xr12","xr13", "xr14", "xr15", "xr16", "xr17", "xr18"
          from "join_result";"""

join_11 = """create table join_result AS select * from "s5" join "r5" on "s5"."fk"= "r5"."rid" order by "rid";"""
NR_11 = """select * from "r5";"""
mu_11 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2")
          from "join_result";"""
data_11 = """select  "xs1", "xs2", "rid", "xr1", "xr2" from "join_result";"""

join_12 = """create table join_result AS select * from "s5" join "r25" on "s5"."fk"= "r25"."rid" order by "rid";"""
NR_12 = """select * from "r25";"""
mu_12 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6")
          from "join_result";"""
data_12 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6" from "join_result";"""

join_13 = """create table join_result AS select * from "s5" join "r10" on "s5"."fk"= "r10"."rid" order by "rid";"""
NR_13 = """select * from "r10";"""
mu_13 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10")
           from "join_result";"""
data_13 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10"
          from "join_result";"""

join_14 = """create table join_result AS select * from "s5" join "r30" on "s5"."fk"= "r30"."rid" order by "rid";"""
NR_14 = """select * from "r30";"""
mu_14 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"),avg("xr11"), avg("xr12"),avg("xr13"), avg("xr14") 
           from "join_result";"""
data_14 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
               "xr11", "xr12","xr13", "xr14"
          from "join_result";"""

join_15 = """create table join_result AS select * from "s5" join "r31" on "s5"."fk"= "r31"."rid" order by "rid";"""
NR_15 = """select * from "r31";"""
mu_15 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"),avg("xr13"), avg("xr14"),
          avg("xr15"), avg("xr16"), avg("xr17"), avg("xr18") from "join_result";"""
data_15 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
           "xr11", "xr12","xr13", "xr14", "xr15", "xr16", "xr17", "xr18"
          from "join_result";"""

join_16 = """create table join_result AS select * from "s13" join "r20" on "s13"."fk"= "r20"."rid" order by "rid";"""
NR_16 = """select * from "r20";"""
mu_16 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2")
          from "join_result";"""
data_16 = """select  "xs1", "xs2", "rid", "xr1", "xr2"
          from "join_result";"""

join_17 = """create table join_result AS select * from "s13" join "r29" on "s13"."fk"= "r29"."rid" order by "rid";"""
NR_17 = """select * from "r29";"""
mu_17 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6")
          from "join_result";"""
data_17 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6"
          from "join_result";"""

join_18 = """create table join_result AS select * from "s13" join "r24" on "s13"."fk"= "r24"."rid" order by "rid";"""
NR_18 = """select * from "r24";"""
mu_18 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10") 
           from "join_result";"""
data_18 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10"
          from "join_result";"""

join_19 = """create table join_result AS select * from "s13" join "r32" on "s13"."fk"= "r32"."rid" order by "rid";"""
NR_19 = """select * from "r32";"""
mu_19 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10") ,avg("xr11"), avg("xr12"),avg("xr13"), avg("xr14") 
           from "join_result";"""
data_19 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
           "xr11", "xr12","xr13", "xr14"
          from "join_result";"""

join_20 = """create table join_result AS select * from "s13" join "r33" on "s13"."fk"= "r33"."rid" order by "rid";"""
NR_20 = """select * from "r33";"""
mu_20 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"),avg("xr13"), avg("xr14"),
          avg("xr15"), avg("xr16"), avg("xr17"), avg("xr18") from "join_result";"""
data_20 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
           "xr11", "xr12","xr13", "xr14", "xr15", "xr16", "xr17", "xr18"
          from "join_result";"""

join_21 = """create table join_result AS select * from "s14" join "r20" on "s14"."fk"= "r20"."rid" order by "rid";"""
NR_21 = """select * from "r20";"""
mu_21 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2")
          from "join_result";"""
data_21 = """select  "xs1", "xs2", "rid", "xr1", "xr2" from "join_result";"""

join_22 = """create table join_result AS select * from "s14" join "r29" on "s14"."fk"= "r29"."rid" order by "rid";"""
NR_22 = """select * from "r29";"""
mu_22 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6")
          from "join_result";"""
data_22 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6" from "join_result";"""

join_23 = """create table join_result AS select * from "s14" join "r24" on "s14"."fk"= "r24"."rid" order by "rid";"""
NR_23 = """select * from "r24";"""
mu_23 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10")
           from "join_result";"""
data_23 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10"
          from "join_result";"""

join_24 = """create table join_result AS select * from "s14" join "r32" on "s14"."fk"= "r32"."rid" order by "rid";"""
NR_24 = """select * from "r32";"""
mu_24 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"),avg("xr11"), avg("xr12"),avg("xr13"), avg("xr14") 
           from "join_result";"""
data_24 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
               "xr11", "xr12","xr13", "xr14"
          from "join_result";"""

join_25 = """create table join_result AS select * from "s14" join "r33" on "s14"."fk"= "r33"."rid" order by "rid";"""
NR_25 = """select * from "r33";"""
mu_25 = """select avg("xs1"), avg("xs2"),avg("xr1"), avg("xr2"),avg("xr3"), avg("xr4"), avg("xr5"), avg("xr6"), 
           avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"),avg("xr13"), avg("xr14"),
          avg("xr15"), avg("xr16"), avg("xr17"), avg("xr18") from "join_result";"""
data_25 = """select  "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
           "xr11", "xr12","xr13", "xr14", "xr15", "xr16", "xr17", "xr18"
          from "join_result";"""

join_sql_list = [join_1, join_2, join_3, join_4, join_5, join_6, join_7, join_8, join_9, join_10,
                 join_11, join_12, join_13, join_14, join_15, join_16, join_17, join_18, join_19, join_20,
                 join_21, join_22, join_23, join_24, join_25]
NR_sql_list = [NR_1, NR_2, NR_3, NR_4, NR_5, NR_6, NR_7, NR_8, NR_9, NR_10, NR_11, NR_12, NR_13, NR_14, NR_15,
               NR_16, NR_17, NR_18, NR_19, NR_20, NR_21, NR_22, NR_23, NR_24, NR_25]
mu_sql_list = [mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7, mu_8, mu_9, mu_10, mu_11, mu_12, mu_13, mu_14, mu_15,
               mu_16, mu_17, mu_18, mu_19, mu_20, mu_21, mu_22, mu_23, mu_24, mu_25]
data_sql_list = [data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9,
                 data_10, data_11, data_12, data_13, data_14, data_15, data_16, data_17, data_18, data_19, data_20,
                 data_21, data_22, data_23, data_24, data_25]
D_list = [4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20]
runtime_list = []
cputime_list = []
fetchdatatime_list = []
calculatetime_list = []
NR_b_list = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]

# runtime_start_1 = runtime_start_2 = runtime_start_3 = runtime_start_4 = runtime_start_5 = 0
# cpu_start_1 = cpu_start_2 = cpu_start_3 = cpu_start_4 = cpu_start_5 = 0
# runtime_start_list = [runtime_start_1, runtime_start_2, runtime_start_3, runtime_start_4, runtime_start_5]
# cpu_start_list = [cpu_start_1, cpu_start_2, cpu_start_3, cpu_start_4, cpu_start_5]
for i in range(20, len(join_sql_list)):
    print("i=", i)
    runtime_start = time.time()
    # cpu_start = time.process_time()
    preparation_start = time.time()
    conn = psycopg2.connect(database="experiment_syn", user="postgres", password="123456", host="localhost",
                            port="5432")

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
