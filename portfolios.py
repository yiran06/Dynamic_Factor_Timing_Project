#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:14:28 2017

@author: kunmingwu
"""
import pandas as pd
import numpy as np
import data_cleaning as DC
import imp

imp.reload(DC)



# notice that PE seems very out of place in terms of return
data = DC.get_all_data()
col = data.columns

#==============================================================================
# TRADING/HOLDING COSTS
#==============================================================================
# please refer to page 5 of the project description
# the following are estimates 

trading_costs = np.array([0.0010, 0.0015, 0.010, 0.0030, 0.0000, 0.0045, 0.0100])
holding_costs = np.array([0.0010, 0.0005, 0.000, 0.0015, 0.0000, 0.0025, 0.0000])/12

#==============================================================================
# EQUAL WEIGHTS BUY AND HOLD PORTFOLIO
#==============================================================================





                         
#==============================================================================
# REFERENCE PORTFOLIO
#==============================================================================

# weights of each corresponding column
weights = np.array([0.5, 0.13, 0.1, 0.025, 0.02, 0.025, 0.1])
# rescaling the weights so that sum is 1, because 
# Aggregate Real Assets and NCREIF are missing
weights = weights * 1/np.sum(weights)


