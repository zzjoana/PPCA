from math import ceil
from src.fppca.FPPCA import FPPCA

import numpy as np
import psycopg2


def fetch_count_mu():
    conn = psycopg2.connect(database="F-test", user="postgres", password="123456", host="localhost", port="5432")
    cursor = conn.cursor()

    sql_NR = """select * from "R";"""
    cursor.execute(sql_NR)
    NR = cursor.rowcount
    # print("NR:", NR)

    sql_NS = """select * from "S";"""
    cursor.execute(sql_NS)
    NS = cursor.rowcount
    # print("NS:", NS)

    sql_muS = """select avg("SID"), avg("XS1"), avg("XS2") from "S";"""
    cursor.execute(sql_muS)
    # fetchone return a tuple
    muS_tuple = cursor.fetchone()
    # print("muS_tuple:\n ", muS_tuple, type(muS_tuple))
    # change tuple to array
    muS = np.array(muS_tuple, dtype=float)
    # the result of round is list then change to array again
    # muS = np.array([round(i, 2) for i in muS_array])
    # how to remove decimal(")??
    # print("muS:\n", muS, muS.dtype)

    # sql_count = """select "FK", count(*) from "S" group by "FK" order by "FK"; """
    # cursor.execute(sql_count)
    # weights = cursor.fetchall()
    # print("weights:\n", weights, type(weights))

    sql_sumR = """select sum("R"."RID" * tmp.CNT),sum("R"."XR1" * tmp.CNT), sum("R"."XR2" * tmp.CNT) from "R"
                 join (select "S"."FK", count(*) as CNT from "S" GROUP BY "S"."FK") as tmp
                      on "R"."RID" = tmp."FK";"""
    cursor.execute(sql_sumR)
    sumR = cursor.fetchone()
    # the type is tuple
    # print("sumR:\n", sumR, type(sumR))
    # change tuple to array and divide NS
    muR = np.array(sumR, dtype=float) / NS
    # print("muR:\n", muR, type(muR))

    cursor.close()
    conn.close()
    return NR, NS, muR, muS


def fetch_data_and_calculate(NR, muR, muS):
    NR_b = 2
    total_batch = ceil(NR / NR_b)
    print("total_batch:", total_batch)
    batch_num = 1

    conn = psycopg2.connect(database="F-test", user="postgres", password="123456", host="localhost", port="5432")
    cursorR = conn.cursor()
    cursorS = conn.cursor()

    sql_R = """select * from "R" order by "RID";"""
    cursorR.execute(sql_R)
    while batch_num <= total_batch:
        print("batch_num:", batch_num)
        R_b_list = cursorR.fetchmany(NR_b)
        # fetchmany return a list
        # print("R_b_list:\n ", R_b_list)
        # change list to array
        R_b_array = np.array(R_b_list)
        # change list to matrix
        data_XR_b = np.mat(R_b_list)
        print("data_XR_b:\n", data_XR_b, type(data_XR_b))
        # array change to list like [1,2]; matrix change to list like [[1], [2]];
        # cannot get the first column directly using
        # list but array, then change array of first column to a list
        R_b_id = R_b_array[:, 0].tolist()
        # print("R_b_id:\n", R_b_id, type(R_b_id))

        sql_S = """select * from "S" where "FK" in %s order by "FK";"""
        cursorS.execute(sql_S, [tuple(R_b_id)])
        S_b_list = cursorS.fetchall()
        # fetchall() return a list
        # print("S_b_list:\n", S_b_list)
        # change list to matrix
        data_XS_b_FK = np.mat(S_b_list) # remove the column of FK
        print("data_XS_b_FK:\n", data_XS_b_FK, type(data_XS_b_FK))

        fppca = FPPCA(latent_dim=2, sigma2=1, max_iter=10, NR_b=NR_b, muS=muS, muR=muR, batch_num=batch_num)
        fppca.fit(data_XS_b_FK, data_XR_b)

        batch_num += 1
    cursorR.close()
    cursorS.close()
    conn.close()


NR, NS, muR, muS = fetch_count_mu()
print("NR:", NR)
print("NS:", NS)
print("muR:\n", muR)
print("muS:\n", muS, muS.dtype)

fetch_data_and_calculate(NR, muR, muS)
