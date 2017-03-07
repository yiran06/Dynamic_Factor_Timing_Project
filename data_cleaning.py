# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:19:33 2017

@author: Shuxin Xu, Kunming Wu
"""

import pandas as pd
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