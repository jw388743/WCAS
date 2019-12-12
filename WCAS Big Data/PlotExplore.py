#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:07:33 2019

@author: james
"""
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


projFld = "/Users/james/Documents/ADEC7430"
codeFld = os.path.join(projFld, "PyCode")
fnsFld = os.path.join(codeFld,"_Functions")
outputFld = os.path.join(projFld, "Output")
rawDataFld = os.path.join(projFld, "RawData")
savedDataFld = os.path.join(projFld, "SavedData")
os.path.exists(outputFld)

#%%
features_raw_train_file = os.path.join(rawDataFld, "dengue_features_train.csv")
labels_raw_train_file = os.path.join(rawDataFld, "dengue_labels_train.csv")
features_raw_train = pd.read_csv(features_raw_train_file)
labels_raw_train = pd.read_csv(labels_raw_train_file)
features_raw_train['cases']=labels_raw_train["total_cases"]

#%%
frt=features_raw_train.copy()
frt_sj=frt.loc[frt['city']=='sj']
#%%
g = sns.PairGrid(frt_sj)
g.map(plt.scatter)
#%%
plot_vars_weather = frt.columns
plot_vars_weather = [i for i in plot_vars_weather if i not in 
                  ['city','year', 'week_num',
                   'weekofyear','week_start_date','ndvi_ne', 
                   'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'cases']]
#%%
g2 = sns.PairGrid(frt, y_vars=['cases'], x_vars=plot_vars_weather, hue='city' )
g2.map(plt.scatter)
#%%
g3 = sns.PairGrid(frt, y_vars=['cases'], x_vars=plot_vars_weather, hue='city' )
g3.map(plt.scatter)
#%%
g4=sns.PairGrid(frt, y_vars=plot_vars_weather, x_vars=['year', 'weekofyear'], hue='city' )
g4.map(plt.scatter)
#%%
station_vars= ['station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c', 'station_precip_mm']
#%%
reanalysis_vars = [i for i in plot_vars_weather if i not in 
                  ['city','year', 'week_num',
                   'weekofyear','week_start_date','ndvi_ne', 
                   'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'cases', 'station_avg_temp_c',
                   'station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c', 'station_precip_mm']]
#%%
ndvi_vars=['ndvi_ne','ndvi_nw', 'ndvi_se', 'ndvi_sw']
#%%
#NDVI Plots. Lets first put the variables on X axis, and cases on Y
ndvi_plt=sns.PairGrid(frt, y_vars=['cases'], x_vars=ndvi_vars, hue='city' )
ndvi_plt.map(plt.scatter)
ndvi_plt.add_legend()
#%%

#Now we will track them over time, withh time on X axis, and NDVI values on Y.

ndvi_plt_time=sns.PairGrid(frt, y_vars=ndvi_vars, x_vars=['year','weekofyear'], hue='city')
ndvi_plt_time.map(plt.scatter)
ndvi_plt_time.add_legend()

#%%

#Now we can repeat for the above defined variables groups, station, and reanalysis. 

station_plt=sns.PairGrid(frt, y_vars=['cases'], x_vars=station_vars, hue='city' )
station_plt.map(plt.scatter)
station_plt.add_legend()
#%%

station_plt_time=sns.PairGrid(frt, y_vars=station_vars, x_vars=['year','weekofyear',], hue='city')
station_plt_time.map(plt.scatter)
station_plt_time.add_legend()

#%%
reanalysis_plt=sns.PairGrid(frt, y_vars=['cases'], x_vars=reanalysis_vars, hue='city')
reanalysis_plt.map(plt.scatter)
reanalysis_plt.add_legend()
#%%
reanalysis_plt_time=sns.PairGrid(frt, y_vars=reanalysis_vars, x_vars= ['year','weekofyear'], hue='city')
reanalysis_plt_time.map(plt.scatter)
reanalysis_plt_time.add_legend()

#%%























