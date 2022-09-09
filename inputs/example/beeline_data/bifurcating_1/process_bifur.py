#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:39:07 2022

@author: cliffzhou
"""
#%%
import pandas as pd
df = pd.read_csv('bifurcating_merged.csv',index_col=0)
df.columns = df.columns.str.lstrip("x_")
df = df.T.add_prefix('cell_')
df.to_csv('bifur_expression.csv')
#%% 
pt0 = pd.read_csv('bifurcating_state_0_pst.csv',index_col=0).add_suffix('0')
pt1 = pd.read_csv('bifurcating_state_1_pst.csv',index_col=0).add_suffix('1')
pt2 = pd.read_csv('bifurcating_state_2_pst.csv',index_col=0).add_suffix('2')
pt = pt0.join(pt1,how = 'outer')
pt = pt.join(pt2,how = 'outer')
#%%
pt.index = df.columns
pt.to_csv('bifur_pst.csv')

#%% generate cluster-specific input files
exp0 = pd.read_csv('bifurcating_state_0.csv',index_col=0)
exp0.columns = exp0.columns.str.lstrip("x_")
exp0 = exp0.T.add_prefix('cell_')
pt0.index = exp0.columns
exp0.to_csv('bifur_exp_0.csv')
pt0.to_csv('bifur_pst_0.csv')
#%% generate cluster-specific input files
exp1 = pd.read_csv('bifurcating_state_1.csv',index_col=0)
exp1.columns = exp1.columns.str.lstrip("x_")
exp1 = exp1.T.add_prefix('cell_')
pt1.index = exp1.columns
exp1.to_csv('bifur_exp_1.csv')
pt1.to_csv('bifur_pst_1.csv')