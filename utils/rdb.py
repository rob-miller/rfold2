# database support

import psycopg2
import pickle

rpVerbose = False


def openDb():
    try:
        conn = psycopg2.connect(
            host="hal.rtm.home", database="rfold", user="rfold", password="rfold"
        )
        """
        cur = conn.cursor()
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

        # close the communication with the PostgreSQL
        cur.close()
        """
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return None


def closeDb(conn):
    if conn is not None:
        conn.close()
        print("Database connection closed.")


def pqry(cur, qry: str):
    global rpVerbose
    try:
        if rpVerbose:
            print(qry)
        cur.execute(qry)
    except (Exception, psycopg2.DatabaseError) as error:
        print(qry)
        print(error)
        raise
    # cur.close()


def pqry1(cur, qry: str):
    global rpVerbose
    pqry(cur, qry)
    try:
        r = cur.fetchone()
    except Exception as e:
        print(qry)
        print(e)
        raise
    if rpVerbose:
        print(f" -> {r}")
    return None if r is None else r[0] if len(r) == 1 else r


def pgetset(cur, qry, insrt, conn=None):
    key = pqry1(cur, qry)
    if key is not None:
        return key
    else:
        # print(insrt)
        # because race
        # insrt2 = f"{insrt} on conflict do nothing"
        rslt = pqry1(cur, insrt)
        if cur.statusmessage == "INSERT 0 1":
            if conn is not None:
                conn.commit()
        else:
            # print(f".{cur.statusmessage}.")
            # no insert means no returned value
            rslt = pqry1(cur, qry)

        return rslt
        # return pqry1(cur, qry)


def pqry1p(cur, qry: str):
    global rpVerbose
    pqry(cur, qry)
    try:
        r = cur.fetchone()
    except Exception as e:
        print(qry)
        print(e)
        raise
    if rpVerbose:
        print(f" -> {r}")
    return None if r is None else pickle.loads(r[0])
