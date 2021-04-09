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


# sql_NR = """select * from "r7";"""
# sql_N = """select * from "s2";"""
# sql_mu = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"),
#              avg("xr3"), avg("xr4"), avg("xr5") , avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"),
#              avg("xr11"), avg("xr12"),  avg("xr13"), avg("xr14"), avg("xr15")from
#              (select * from "s2" join "r7" on "s2"."fk"= "r7"."rid") as tmp;"""
# sql_R = """select "rid" from "r7" order by "rid";"""
# sql_b = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
#             "xr11", "xr12", "xr13", "xr14", "xr15" from (select * from "s2" where "fk" in %s) as tmp1
#             join (select * from "r7" where "rid" in %s ) as tmp2 on tmp1."fk"=tmp2."rid" order by "rid";"""
# D = 17
#
# NR_1 = """select * from "r1";"""
# N_1 = """select * from "s1";"""
# mu_1 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2") from
#        (select * from "s1" join "r1" on "s1"."fk"= "r1"."rid") as tmp;"""
# R_1 = """select "rid" from "r1" order by "rid";"""
# b_1 = """select "xs1", "xs2", "rid", "xr1", "xr2"  from "s1" as tmp1,"r1" as tmp2
#          where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid";"""
NR_1 = """select * from "r3";"""
N_1 = """select * from "s3";"""
mu_1 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2")
         from (select * from "s3" join "r3" on "s3"."fk"= "r3"."rid") as tmp;"""
R_1 = """select "rid" from "r3" ;"""
b_1 = """select "xs1", "xs2", "rid", "xr1", "xr2"
        from "s3" as tmp1,"r3" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_2 = """select * from "r11";"""
N_2 = """select * from "s3";"""
mu_2 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6")from (select * from "s3" join "r11" on "s3"."fk"= "r11"."rid") as tmp;"""
R_2 = """select "rid" from "r11" ;"""
b_2 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5", "xr6"
        from "s3" as tmp1,"r11" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_3 = """select * from "r8";"""
N_3 = """select * from "s3";"""
mu_3 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10")
           from (select * from "s3" join "r8" on "s3"."fk"= "r8"."rid") as tmp;"""
R_3 = """select "rid" from "r8" ;"""
b_3 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10"
        from "s3" as tmp1,"r8" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_4 = """select * from "r12";"""
N_4 = """select * from "s3";"""
mu_4 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"), 
          avg("xr14") from (select * from "s3" join "r12" on "s3"."fk"= "r12"."rid") as tmp;"""
R_4 = """select "rid" from "r12" ;"""
b_4 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
         "xr11","xr12", "xr13","xr14" from "s3" as tmp1,"r12" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_5 = """select * from "r13";"""
N_5 = """select * from "s3";"""
mu_5 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"),
          avg("xr14"), avg("xr15"), avg("xr16"), avg("xr17"), avg("xr18")from 
          (select * from "s3" join "r13" on "s3"."fk"= "r13"."rid") as tmp;"""
R_5 = """select "rid" from "r13";"""
b_5 = """select "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
        "xr11", "xr12","xr13", "xr14", "xr15", "xr16", "xr17", "xr18"
         from "s3" as tmp1,"r13" as tmp2
         where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_6 = """select * from "r1";"""
N_6 = """select * from "s1";"""
mu_6 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2")
         from (select * from "s1" join "r1" on "s1"."fk"= "r1"."rid") as tmp;"""
R_6 = """select "rid" from "r1" ;"""
b_6 = """select "xs1", "xs2", "rid", "xr1", "xr2"
        from "s1" as tmp1,"r1" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_7 = """select * from "r14";"""
N_7 = """select * from "s1";"""
mu_7 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6") from  (select * from "s1" join "r14" on "s1"."fk"= "r14"."rid") as tmp;"""
R_7 = """select "rid" from "r14";"""
b_7 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6"
        from "s1" as tmp1,"r14" as tmp2
       where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_8 = """select * from "r6";"""
N_8 = """select * from "s1";"""
mu_8 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10")
           from (select * from "s1" join "r6" on "s1"."fk"= "r6"."rid") as tmp;"""
R_8 = """select "rid" from "r6" ;"""
b_8 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10"
        from "s1" as tmp1,"r6" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_9 = """select * from "r15";"""
N_9 = """select * from "s1";"""
mu_9 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"),
          avg("xr14")from (select * from "s1" join "r15" on "s1"."fk"= "r15"."rid") as tmp;"""
R_9 = """select "rid" from "r15" ;"""
b_9 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
          "xr11", "xr12", "xr13", "xr14"
           from "s1" as tmp1,"r15" as tmp2
       where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_10 = """select * from "r16";"""
N_10 = """select * from "s1";"""
mu_10 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"),
          avg("xr14"),avg("xr15"), avg("xr16"), avg("xr17"), avg("xr18")
          from (select * from "s1" join "r16" on "s1"."fk"= "r16"."rid") as tmp;"""
R_10 = """select "rid" from "r16" ;"""
b_10 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
          "xr11", "xr12", "xr13", "xr14","xr15", "xr16", "xr17", "xr18"
          from "s1" as tmp1,"r16" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_11 = """select * from "r5";"""
N_11 = """select * from "s5";"""
mu_11 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2")
         from (select * from "s5" join "r5" on "s5"."fk"= "r5"."rid") as tmp;"""
R_11 = """select "rid" from "r5" ;"""
b_11 = """select "xs1", "xs2", "rid", "xr1", "xr2"
        from "s5" as tmp1,"r5" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_12 = """select * from "r25";"""
N_12 = """select * from "s5";"""
mu_12 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6")from (select * from "s5" join "r25" on "s5"."fk"= "r25"."rid") as tmp;"""
R_12 = """select "rid" from "r25" ;"""
b_12 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5", "xr6"
        from "s5" as tmp1,"r25" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_13 = """select * from "r10";"""
N_13 = """select * from "s5";"""
mu_13 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10")
           from (select * from "s5" join "r10" on "s5"."fk"= "r10"."rid") as tmp;"""
R_13 = """select "rid" from "r10" ;"""
b_13 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10"
        from "s5" as tmp1,"r10" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_14 = """select * from "r30";"""
N_14 = """select * from "s5";"""
mu_14 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"), 
          avg("xr14") from (select * from "s5" join "r30" on "s5"."fk"= "r30"."rid") as tmp;"""
R_14 = """select "rid" from "r30" ;"""
b_14 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
         "xr11","xr12", "xr13","xr14" from "s5" as tmp1,"r30" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_15 = """select * from "r31";"""
N_15 = """select * from "s5";"""
mu_15 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"),
          avg("xr14"), avg("xr15"), avg("xr16"), avg("xr17"), avg("xr18")from 
          (select * from "s5" join "r31" on "s5"."fk"= "r31"."rid") as tmp;"""
R_15 = """select "rid" from "r31";"""
b_15 = """select "xs1", "xs2", "rid", "xr1", "xr2","xr3", "xr4", "xr5", "xr6", "xr7", "xr8", "xr9", "xr10",
        "xr11", "xr12","xr13", "xr14", "xr15", "xr16", "xr17", "xr18"
         from "s5" as tmp1,"r31" as tmp2
         where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_16 = """select * from "r20";"""
N_16 = """select * from "s13";"""
mu_16 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2")
         from (select * from "s13" join "r20" on "s13"."fk"= "r20"."rid") as tmp;"""
R_16 = """select "rid" from "r20" ;"""
b_16 = """select "xs1", "xs2", "rid", "xr1", "xr2"
        from "s13" as tmp1,"r20" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_17 = """select * from "r29";"""
N_17 = """select * from "s13";"""
mu_17 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6") from  (select * from "s13" join "r29" on "s13"."fk"= "r29"."rid") as tmp;"""
R_17 = """select "rid" from "r29";"""
b_17 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6"
        from "s13" as tmp1,"r29" as tmp2
       where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_18 = """select * from "r24";"""
N_18 = """select * from "s13";"""
mu_18 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10")
           from (select * from "s13" join "r24" on "s13"."fk"= "r24"."rid") as tmp;"""
R_18 = """select "rid" from "r24" ;"""
b_18 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10"
        from "s13" as tmp1,"r24" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_19 = """select * from "r32";"""
N_19 = """select * from "s13";"""
mu_19 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"),
          avg("xr14")from (select * from "s13" join "r32" on "s13"."fk"= "r32"."rid") as tmp;"""
R_19 = """select "rid" from "r32" ;"""
b_19 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
          "xr11", "xr12", "xr13", "xr14"
           from "s13" as tmp1,"r32" as tmp2
       where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_20 = """select * from "r33";"""
N_20 = """select * from "s13";"""
mu_20 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"),
          avg("xr14"),avg("xr15"), avg("xr16"), avg("xr17"), avg("xr18")
          from (select * from "s13" join "r33" on "s13"."fk"= "r33"."rid") as tmp;"""
R_20 = """select "rid" from "r33" ;"""
b_20 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
          "xr11", "xr12", "xr13", "xr14","xr15", "xr16", "xr17", "xr18"
          from "s13" as tmp1,"r33" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_21 = """select * from "r20";"""
N_21 = """select * from "s14";"""
mu_21 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2")
         from (select * from "s14" join "r20" on "s14"."fk"= "r20"."rid") as tmp;"""
R_21 = """select "rid" from "r20" ;"""
b_21 = """select "xs1", "xs2", "rid", "xr1", "xr2"
        from "s14" as tmp1,"r20" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_22 = """select * from "r29";"""
N_22 = """select * from "s14";"""
mu_22 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6") from  (select * from "s14" join "r29" on "s14"."fk"= "r29"."rid") as tmp;"""
R_22 = """select "rid" from "r29";"""
b_22 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6"
        from "s14" as tmp1,"r29" as tmp2
       where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_23 = """select * from "r24";"""
N_23 = """select * from "s14";"""
mu_23 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10")
           from (select * from "s14" join "r24" on "s14"."fk"= "r24"."rid") as tmp;"""
R_23 = """select "rid" from "r24" ;"""
b_23 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10"
        from "s14" as tmp1,"r24" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_24 = """select * from "r32";"""
N_24 = """select * from "s14";"""
mu_24 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"),
          avg("xr14")from (select * from "s14" join "r32" on "s14"."fk"= "r32"."rid") as tmp;"""
R_24 = """select "rid" from "r32" ;"""
b_24 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
          "xr11", "xr12", "xr13", "xr14"
           from "s14" as tmp1,"r32" as tmp2
       where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""

NR_25 = """select * from "r33";"""
N_25 = """select * from "s14";"""
mu_25 = """select  avg("xs1"), avg("xs2"), avg("xr1"), avg("xr2"), avg("xr3"), avg("xr4"), avg("xr5"),
          avg("xr6"), avg("xr7"), avg("xr8"), avg("xr9"), avg("xr10"), avg("xr11"), avg("xr12"), avg("xr13"),
          avg("xr14"),avg("xr15"), avg("xr16"), avg("xr17"), avg("xr18")
          from (select * from "s14" join "r33" on "s14"."fk"= "r33"."rid") as tmp;"""
R_25 = """select "rid" from "r33" ;"""
b_25 = """select "xs1", "xs2", "rid", "xr1", "xr2", "xr3", "xr4", "xr5" , "xr6", "xr7", "xr8", "xr9", "xr10",
          "xr11", "xr12", "xr13", "xr14","xr15", "xr16", "xr17", "xr18"
          from "s14" as tmp1,"r33" as tmp2
        where "fk" in %s and tmp1."fk"=tmp2."rid" order by "rid", "sid";"""
NR_sql_list = [NR_1, NR_2, NR_3, NR_4, NR_5, NR_6, NR_7, NR_8, NR_9, NR_10, NR_11, NR_12, NR_13, NR_14, NR_15, NR_16,
               NR_17, NR_18, NR_19, NR_20, NR_21, NR_22, NR_23, NR_24, NR_25]
N_sql_list = [N_1, N_2, N_3, N_4, N_5, N_6, N_7, N_8, N_9, N_10, N_11, N_12, N_13, N_14, N_15, N_16, N_17, N_18, N_19,
              N_20, N_21, N_22, N_23, N_24, N_25]
mu_sql_list = [mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7, mu_8, mu_9, mu_10, mu_11, mu_12, mu_13, mu_14, mu_15, mu_16,
               mu_17, mu_18, mu_19, mu_20, mu_21, mu_22, mu_23, mu_24, mu_25]
R_sql_list = [R_1, R_2, R_3, R_4, R_5, R_6, R_7, R_8, R_9, R_10, R_11, R_12, R_13, R_14, R_15, R_16, R_17, R_18, R_19,
              R_20, R_21, R_22, R_23, R_24, R_25]
b_sql_list = [b_1, b_2, b_3, b_4, b_5, b_6, b_7, b_8, b_9, b_10, b_11, b_12, b_13, b_14, b_15, b_16, b_17, b_18, b_19,
              b_20, b_21, b_22, b_23, b_24, b_25]

D_list = [4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20]
runtime_list = []
cputime_list = []
fetchdatatime_list = []
calculatetime_list = []
NR_b_list = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]

for i in range(20, len(NR_sql_list)):
    print("i=", i)
    runtime_start = time.time()
    # cpu_start = time.process_time()
    preparation_start = time.time()

    conn = psycopg2.connect(database="experiment_syn", user="postgres", password="123456", host="localhost",
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
