# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 15:55:41 2018

@author: VStore
"""

import numpy as np
import pandas as pd

def columns_not_in_another(df1,df2):
    a = [i for i in df1.columns.to_list() if i not in df2.columns ]
    b = [i for i in df2.columns.to_list() if i not in df1.columns ]
    print('DataFrame 1 shape {} DataFrame 2 shape {}'.format(df1.shape,df2.shape))
    if len(a) > 0:
        print('Columns in 1 but not in 2:')
        print(a)
    else:
        print('Columns in 1 are all in 2')
    if len(b) > 0:
        print('Columns in 2 but not in 1:')
        print(b)
    else:
        print('Columns in 2 are all in 1')

def common_columns(df1,df2):
    print('DataFrame 1 shape {} DataFrame 2 shape {}'.format(df1.shape,df2.shape))
    print('Columns in both DataFrames:')
    print([i for i in df1.columns.to_list() if i in df2.columns ])
    
def empty_row_table(df):
    df_index = df.runnerId
    df_nul = df.isnull().sum(axis=1)
    len_row = len(df.columns)
    df_null_percent = 100*df_nul/len_row
    
    df_result = pd.concat([df_index, df_null_percent], axis=1)
    df_result = df_result.set_index('runnerId')
    df_result.rename(columns = {0 : 'runnerId', 1 : 'nullPercent'},inplace=True)
    df_result.columns  = ['runnerId','nullPercent']



# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent,df.dtypes], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'dtype'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
    
    
def unique_values_table(df):       
        
        uniq =  df.apply(pd.Series.nunique, axis = 0)
        
        uniq = pd.DataFrame({'column':uniq.index, 'count':uniq.values, 'dtype':df.dtypes})

        return uniq.sort_values('count', ascending=False)
    
    
def reorder_columns(df):
    uni_df = unique_values_table(df)

    order = []
    for col in uni_df.column.to_list():
        if 'id' in str.lower(col):
            order.append(col)

    for col in uni_df.column.to_list():
        if col in ['kmTime','result','position','prediction','order','disqualificationCode']:
            order.append(col)


    for col in uni_df.column.to_list():
        if col not in order:
            order.append(col)

    del uni_df
    return df[order].copy()
    
def listify_colmuns(df):
    print('[')
    for column in df.columns:
        print('\''+column+'\',')
        
        
    print(']')
    
    
    