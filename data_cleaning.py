# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:19:33 2017

@author: Shuxin Xu
"""

import pandas as pd
import matplotlib.pyplot as plt


sheet_name=['MSCI ACWI','Barclays US Agg','HFRI FOF','Barclays TIP US Index','JPM EMBI','NCREIF']

for sheet in sheet_name:
    data=pd.read_excel('factor_timing_project_data_cleaned.xlsx',sheetname=sheet)
    data=data.iloc[:,0:2]
    data['Date']=pd.to_datetime(data['Date'],infer_datetime_format=True)
    fig=plt.figure()
    plt.plot(data['Date'],data['Last_Price'])
    plt.legend([sheet])