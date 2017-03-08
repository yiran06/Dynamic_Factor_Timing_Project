# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:19:33 2017

@author: Shuxin Xu, Kunming Wu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#==============================================================================
# PLOTING
#==============================================================================
# plot reference portfolio component price time series
# remove 'NCREIF', because it has only quarterly data
def plot_data_1():
    sheet_name=['MSCI ACWI','Barclays US Agg','HFRI FOF','ML US HY Cash Pay','Barclays TIP US Index','JPM EMBI']
    for sheet in sheet_name:
        data=pd.read_excel('./data/factor_timing_project_data_cleaned.xlsx',sheetname=sheet)
        data=data.iloc[:,0:2]
        data['Date']=pd.to_datetime(data['Date'],infer_datetime_format=True)
        plt.plot(data['Date'],data['Last_Price'])
        plt.legend([sheet])

def plot_data_2():
    # plot PE return        
    sheet_name=['PE']
    for sheet in sheet_name:
        data=pd.read_excel('./data/factor_timing_project_data_cleaned.xlsx',sheetname=sheet)
        data=data.iloc[:,0:3]
        data['EndDate']=pd.to_datetime(data['EndDate'],infer_datetime_format=True)
        plt.plot(data['EndDate'],data['Monthly_Return'])
        plt.legend([sheet])

#==============================================================================
# COMBINE DATAFRAME
#==============================================================================
def to_monthly_return(df):
    # make sure the order is correct, the first row should be the latest data
    if df['Date'][0] < df['Date'][1]:
        df = df[::-1]
    df.index = pd.to_datetime(df['Date'])
    df = df.groupby([lambda x: x.year, lambda x: x.month]).first()
    # reset index to include only the last day of month
    # use month end data here
    df.index = pd.to_datetime(df['Date'])
    df = df['Last_Price']
    ret = df.diff()/df.shift(1)
    ret.index = ret.index.map(lambda x:x.strftime('%Y-%m'))
    return ret[1:]

def get_all_data():
    sheet_name=['MSCI ACWI','Barclays US Agg','HFRI FOF','ML US HY Cash Pay','Barclays TIP US Index','JPM EMBI']
    
    df_arr = []
    for sheet in sheet_name:
        df = pd.read_excel('./data/factor_timing_project_data_cleaned.xlsx', sheetname = sheet)
        df_arr.append(to_monthly_return(df))
    combined_df = pd.DataFrame(df_arr).T
    combined_df.columns = sheet_name
    
    # special treatment for PE
    PE_data = pd.read_excel('./data/factor_timing_project_data_cleaned.xlsx', sheetname = 'PE')
    PE_data['EndDate'] = pd.to_datetime(PE_data['EndDate'])
    PE_data.index = PE_data['EndDate']
    PE_data = PE_data['Monthly_Return']/100
    PE_data.index = PE_data.index.map(lambda x:x.strftime('%Y-%m'))
    combined_df['PE'] = PE_data
    
    valid_rows = [True] * len(combined_df)
    for i,col in enumerate(combined_df.columns):
        valid_rows &= ~np.isnan(combined_df[col])
    valid_rows
    
    combined_df = combined_df.ix[valid_rows]
    return combined_df
   
   
   
def get_all_data2():
    file_name=['Equity_russel_3000','Equity_global_MSCI_ACWI',\
    'Bond_US_Barclay_ETF','Bond_US_Barclay_TIPS','Bond_US_HighYield_bond_Index',\
    'IYR','FOF_Fund of Fund']
    df_arr = []
    for file in file_name:
        df = pd.read_excel('./data/asset/'+file+'.xlsx')
        df_arr.append(to_monthly_return(df))
    combined_df = pd.DataFrame(df_arr).T
    combined_df.columns = file_name
    
    # special treatment for PE
    PE_data = pd.read_excel('./data/asset/UCRP PE.xlsx')
    PE_data['EndDate'] = pd.to_datetime(PE_data['EndDate'])
    PE_data.index = PE_data['EndDate']
    PE_data = PE_data['Monthly_Return']/100
    PE_data.index = PE_data.index.map(lambda x:x.strftime('%Y-%m'))
    combined_df['PE'] = PE_data
    
    valid_rows = [True] * len(combined_df)
    for i,col in enumerate(combined_df.columns):
        valid_rows &= ~np.isnan(combined_df[col])
    valid_rows
    
    combined_df = combined_df.ix[valid_rows]
    return combined_df