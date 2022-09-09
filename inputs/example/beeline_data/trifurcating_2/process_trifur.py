#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:39:07 2022

@author: cliffzhou
"""
#%%
import pandas as pd
df = pd.read_csv('trifurcating_merged.csv',index_col=0)
df = df.T.add_prefix('cell_')
df.to_csv('trifur_expression.csv')
#%% 
pt0 = pd.read_csv('trifurcating_state_0_pst.csv',index_col=0).add_suffix('0')
pt1 = pd.read_csv('trifurcating_state_1_pst.csv',index_col=0).add_suffix('1')
pt2 = pd.read_csv('trifurcating_state_2_pst.csv',index_col=0).add_suffix('2')
pt3 = pd.read_csv('trifurcating_state_3_pst.csv',index_col=0).add_suffix('3')

pt = pt0.join(pt1,how = 'outer')
pt = pt.join(pt2,how = 'outer')
pt = pt.join(pt3,how = 'outer')
#%%
pt.index = df.columns
pt.to_csv('trifur_pst.csv')

#%% generate cluster-specific input files
exp0 = pd.read_csv('trifurcating_state_0.csv',index_col=0)
exp0 = exp0.T.add_prefix('cell_')
pt0.index = exp0.columns
exp0.to_csv('trifur_exp_0.csv')
pt0.to_csv('trifur_pst_0.csv')
#%% generate cluster-specific input files
exp1 = pd.read_csv('trifurcating_state_1.csv',index_col=0)
exp1 = exp1.T.add_prefix('cell_')
pt1.index = exp1.columns
exp1.to_csv('trifur_exp_1.csv')
pt1.to_csv('trifur_pst_1.csv')
#%% generate cluster-specific input files
exp2 = pd.read_csv('trifurcating_state_2.csv',index_col=0)
exp2 = exp2.T.add_prefix('cell_')
pt2.index = exp2.columns
exp2.to_csv('trifur_exp_2.csv')
pt2.to_csv('trifur_pst_2.csv')
#%% generate cluster-specific input files
exp3 = pd.read_csv('trifurcating_state_3.csv',index_col=0)
exp3 = exp3.T.add_prefix('cell_')
pt3.index = exp3.columns
exp3.to_csv('trifur_exp_3.csv')
pt3.to_csv('trifur_pst_3.csv')
