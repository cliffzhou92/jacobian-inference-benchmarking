#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:39:07 2022

@author: cliffzhou
"""
#%%
import pandas as pd
df = pd.read_csv('cycle.csv',index_col=0)
df.columns = df.columns.str.lstrip("x_")
df = df.T.add_prefix('cell_')
df.to_csv('cycle_expression.csv')
#%% 
pt = pd.read_csv('cycle_pst.csv',index_col=0).squeeze()
pt = pt.add_prefix('cell_')
pt.to_csv('cycle_pst.csv')
