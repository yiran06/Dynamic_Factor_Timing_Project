# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:19:33 2017

@author: Shuxin Xu, Kunming Wu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##plot reference portfolio component price time series
sheet_name=['MSCI ACWI','Barclays US Agg','HFRI FOF','ML US HY Cash Pay','Barclays TIP US Index','JPM EMBI','NCREIF']
for sheet in sheet_name:
    data=pd.read_excel('./data/factor_timing_project_data_cleaned.xlsx',sheetname=sheet)
    data=data.iloc[:,0:2]
    data['Date']=pd.to_datetime(data['Date'],infer_datetime_format=True)
    fig=plt.figure()
    plt.plot(data['Date'],data['Last_Price'])
    plt.legend([sheet])

    
##plot PE return        
sheet_name=['PE']
for sheet in sheet_name:
    data=pd.read_excel('./data/factor_timing_project_data_cleaned.xlsx',sheetname=sheet)
    data=data.iloc[:,0:3]
    data['EndDate']=pd.to_datetime(data['EndDate'],infer_datetime_format=True)
    fig=plt.figure()
    plt.plot(data['EndDate'],data['Monthly_Return'])
    plt.legend([sheet])


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


df_arr = []
for sheet in sheet_name:
    df = pd.read_excel('./data/factor_timing_project_data_cleaned.xlsx', sheetname = sheet)
    df_arr.append(to_monthly_return(df))
combined_df = pd.DataFrame(df_arr).T
combined_df.columns = sheet_name

for i,col in enumerate(combined_df.columns):
    
    