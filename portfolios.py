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
from cvxpy import *
    
imp.reload(DC)
plt.style.use('ggplot')

# notice that PE seems very out of place in terms of return
data = pd.read_csv('./data/asset_return.csv')
#data.drop(['PE'], axis=1, inplace=True)
col = data.columns
data['Date']=pd.to_datetime(data['Date'])
data.set_index(data['Date'],inplace=True)
del data['Date']




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
def portfolio(data,weights,legend):    
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
    plt.plot(pd.to_datetime(data.index), p_sum, label=legend)
    #plt.plot(pd.to_datetime(data.index)[1:], p_ret, label='Monthly Return')
    plt.legend(loc='best')
    return p_ret




def risk_parity(freq=12):
    weights=[list(data.iloc[range(0,12),:].apply(lambda x: 1/np.std(x),axis=0))]
    for i in range(1,len(data)-12):
        weights=np.vstack((weights,list(data.iloc[range(i,i+12),:].apply(lambda x: 1/np.std(x),axis=0))))
    return weights




def risk_parity2(alpha=0.8):
    weights=[]
    for j in range(data.shape[0]):
        ## calculate exponential weights
        exp_w=np.array([(1-alpha)*alpha**i for i in range(j,-1,-1)])
        exp_w=exp_w/sum(exp_w)
        exp_ret=np.array(data.iloc[0:(j+1),:])
        w=[]        
        for i in range(exp_ret.shape[1]):
            ##calculate exponential weighted return            
            exp_ret[:,i]=exp_ret[:,i]*exp_w
            exp_var=sum((data.iloc[0:(j+1),i]-sum(exp_ret[:,i]))**2*exp_w)
            w.append(1/np.sqrt(exp_var))
        weights.append(w)
    weights=np.array(weights)
    weights = weights[12:,:]
    return weights
    
    
    


def mean_variance_model(asset_alpha, asset_cov, lam = 10):  
    # cvxopt   
    N_asset = asset_cov.shape[0]
    P = matrix(np.array(asset_cov)) * lam
    q = matrix(-np.array(asset_alpha))
    G = matrix(-np.identity(N_asset))
    h = matrix(np.zeros( (N_asset,1)))
    A = matrix(1.0 + np.zeros((1,N_asset)))
    b = matrix(1.0)

    sol = solvers.qp(P, q, G, h, A, b)

    df_sol = pd.Series(data = np.array(sol['x']).flatten(), index = asset_alpha.index)
    return df_sol




def model0(r,c,lbd):
    r=np.array(r)
    n=len(r)
    x=Variable(n)
    p=Problem(Maximize(r*x-lbd*(quad_form(x, c))))
    p.solve()
    return x.value

def model1(r,c,lbd,w0):
    r=np.array(r)
    w0=np.array(w0)
    n=len(r)
    x=Variable(n)
    p=Problem(Maximize(r*x-lbd*(quad_form(x, c))-trading_costs*(w0-x)))
    p.solve()
    return x.value
    
    
    
def m_v(data,lbd):
    weights=[]
    for i in range(data.shape[0]-12):
        if i==0:
           alpha=data.iloc[i+11,:]
           cov=np.cov(data.iloc[i:i+12,:].T) 
           result=model0(alpha,cov,lbd)
           weights.append(result)
        alpha=data.iloc[i+11,:]
        cov=np.cov(data.iloc[i:i+12,:].T)
        #result=model1(alpha,cov,1000,weights[i-1])
        result=model0(alpha,cov,lbd)
        weights.append(np.array(result))
    weights=np.array(weights)
    weights=weights.reshape(data.shape[0]-11,data.shape[1])
    return np.array(weights)
    
  
    

def mv_portfolio(data,legend):  
    weights=[]
    ##the first alpha and cov
    alpha=data.iloc[11,:]
    cov=np.cov(data.iloc[0:12,:].T) 
    result=model0(alpha,cov,lbd)
    weights.append(result)
    p0 = 1 * weights[0]/np.sum(weights[0]) # initial dollar value 
    p_sum = [1]
    for i in range(1,len(data)-12):
        alpha=data.iloc[i+11,:]
        cov=np.cov(data.iloc[i:i+12,:].T)
        weight=model1(alpha,cov,lbd,weights[0])
        p1 = np.multiply(p0, 1 + data.ix[i] - holding_costs) 
        diff = np.abs(p1 - weight * np.sum(p1))
        p1 = weight * (np.sum(p1) - np.dot(diff, trading_costs))
        p_sum.append(np.sum(p1))
        p0=p1
    #p_ret = (pd.Series(p_sum).diff()/pd.Series(p_sum).shift(1))[1:]  
    p_sum=pd.Series(p_sum)
    p_ret =  np.log(p_sum)-np.log(p_sum).shift(1)
    p_ret=p_ret[1:]    
    plt.plot(pd.to_datetime(data.index), p_sum, label=legend)
    #plt.plot(pd.to_datetime(data.index)[1:], p_ret, label='Monthly Return')
    plt.legend(loc='best')
    return p_ret
    
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


# simple risk parity
portfolio(data.iloc[12:,:],risk_parity(),'simple risk parity')
# expoentially weighted risk parity
portfolio(data.iloc[12:,:],risk_parity2(0.94),'ewm risk parity alpha=0.94')
# expoentially weighted risk parity
portfolio(data.iloc[12:,:],risk_parity2(0.8),'ewm risk parity alpha=0.8')


fig=plt.figure()
for i in range(data.shape[1]):
    plt.plot(np.cumprod(1+data.iloc[:,i]))
plt.legend(data.columns,fontsize=5,loc=0)
plt.savefig('asset return',dpi=200)


#mean variance
weight1=m_v(data,1000000)
weight2=m_v(data,10)
portfolio(data.iloc[11:,:],weight1,'naive mean variance lambda=10^6')
portfolio(data.iloc[11:,:],weight2,'naive mean variance lambda=10')