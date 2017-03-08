#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:14:28 2017

@author: kunmingwu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_cleaning as DC
import imp

imp.reload(DC)
plt.style.use('ggplot')

# notice that PE seems very out of place in terms of return
data = DC.get_all_data()
#data.drop(['PE'], axis=1, inplace=True)
col = data.columns

def buy_and_hold(weights):
    weights = weights * 1/np.sum(weights)    
    portfolio = 1 * weights # initial dollar value 
    p_sum = []
    for i in range(len(data)):
        # monthly return less holding costs
        portfolio = np.multiply(portfolio, 1 + data.ix[i] - holding_costs)     
        p_sum.append(np.sum(portfolio))
    p_ret = (pd.Series(p_sum).diff()/pd.Series(p_sum).shift(1))[1:]
        
    plt.plot(pd.to_datetime(data.index), p_sum, label='Portfolio Value')
    #plt.plot(pd.to_datetime(data.index)[1:], p_ret, label='Monthly Return')
    plt.legend(loc='best')

def monthly_rebalance(weights):
    weights = weights * 1/np.sum(weights)
    portfolio = 1 * weights # initial dollar value 
    p_sum = []
    for i in range(len(data)):
        # monthly return less holding costs
        portfolio = np.multiply(portfolio, 1 + data.ix[i] - holding_costs) 
        diff = np.abs(portfolio - weights * np.sum(portfolio))
        portfolio = weights * (np.sum(portfolio) - np.dot(diff, trading_costs))
        p_sum.append(np.sum(portfolio))
    p_ret = (pd.Series(p_sum).diff()/pd.Series(p_sum).shift(1))[1:]
        
    plt.plot(pd.to_datetime(data.index), p_sum, label='Portfolio Value')
    #plt.plot(pd.to_datetime(data.index)[1:], p_ret, label='Monthly Return')
    plt.legend(loc='best')


###inputs: weights is a matrix, row: end of period date, col: asset weight
###inputs: data is a dataframe, row: end of period date, col: asset return 
def portfolio(data,weights):    
    p0 = 1 * weights[0]/np.sum(weights[0]) # initial dollar value 
    p_sum = [1]
    for i in range(1,len(data)):
        # monthly return less holding costs
        weight = weights[i] * 1/np.sum(weights[i])
        p1 = np.multiply(p0, 1 + data.ix[i] - holding_costs) 
        diff = np.abs(p1 - weight * np.sum(p1))
        p1 = weight * (np.sum(p1) - np.dot(diff, trading_costs))
        p_sum.append(np.sum(p1))
        p0=p1
    #p_ret = (pd.Series(p_sum).diff()/pd.Series(p_sum).shift(1))[1:]  
    p_sum=pd.Series(p_sum)
    p_ret =  np.log(p_sum)-np.log(p_sum).shift(1)
    p_ret=p_ret[1:]    
    plt.plot(pd.to_datetime(data.index), p_sum, label='Portfolio Value')
    #plt.plot(pd.to_datetime(data.index)[1:], p_ret, label='Monthly Return')
    plt.legend(loc='best')
    return p_ret


def risk_parity(freq=12):
    weights=[list(data.iloc[range(0,12),:].apply(lambda x: np.std(x),axis=0))]
    for i in range(1,len(data)-12):
        weights=np.vstack((weights,list(data.iloc[range(i,i+12),:].apply(lambda x: np.std(x),axis=0))))

    
#==============================================================================
# TRADING/HOLDING COSTS
#==============================================================================
# please refer to page 5 of the project description
# the following are estimates 

trading_costs = np.array([0.0005, 0.0010, 0.0015, 0.0000, 0.0030, 0.0040, 0.0100,0.0100])
holding_costs = np.array([0.0000, 0.0010, 0.0005, 0.0000, 0.0015, 0.0025, 0.0000,0.0000])/12

                         
# 60/40 (BUY AND HOLD)
buy_and_hold(np.array([0.6, 0.4, 0, 0, 0, 0, 0]))
# 60/40 (MONTHLY_REBALANCE)
monthly_rebalance(np.array([0.6, 0.4, 0, 0, 0, 0, 0]))

                         
# EQUAL WEIGHTS PORTFOLIO (BUY AND HOLD)
buy_and_hold(np.array([1/7]*7))
# EQUAL WEIGHTS PORTFOLIO (MONTHLY_REBALANCE)
monthly_rebalance(np.array([1/7]*7))


# UCRP (BUY AND HOLD)
buy_and_hold(np.array([0.5, 0.13, 0.1, 0.025, 0.02, 0.025, 0.1]))
# UCRP (MONTHLY_REBALANCE)
monthly_rebalance(np.array([0.5, 0.13, 0.1, 0.025, 0.02, 0.025, 0.1]))









