#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import sqlite3
import pandas as pd


def get_new_cursor(conn):
    #make it a row factory
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    return cur
    
def get_connection_and_cursor(db_name, read_only=True):
    if read_only:
        conn = sqlite3.connect('file:{}?mode=ro'.format(db_name), uri=True)
    else:
        conn = sqlite3.connect(db_name)
    
    #make it a row factory
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    return conn, cur


def get_df_from_table(table_name, conn):
    return pd.read_sql_query("SELECT * FROM {}".format(table_name), conn)


def add_new_column(conn, target_table, new_column_name, new_column_type = "TEXT"):
    '''
    adding a new column; the connection should be pointing to the desired DB
    '''
    add_statement = """ALTER TABLE {} 
                        ADD COLUMN {} {};""".format(target_table, 
                                                    new_column_name,
                                                    new_column_type)
    
    try:
        conn.execute(add_statement)
        conn.commit()
    except sqlite3.OperationalError as e:
        print("Column nod added!")
        print(e) 
    
    return

def create_index_on_column(conn, target_table, to_index_column, index_name=None):
    
    create_index_statement = """CREATE INDEX {} ON {} ({})""".format(index_name, 
                                                                    target_table, 
                                                                    to_index_column)
    conn.execute(create_index_statement)
    conn.commit()
    
    return
    
    
    
