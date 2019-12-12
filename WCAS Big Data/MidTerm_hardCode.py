#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:04:44 2019

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
raw_features_train_file = os.path.join(rawDataFld, "dengue_features_train.csv")
raw_labels_train_file = os.path.join(rawDataFld, "dengue_labels_train.csv")

features_raw_train = pd.read_csv(raw_features_train_file)
labels_raw_train = pd.read_csv(raw_labels_train_file)

#Join cases of fever (last column of labels) to feature df
features_raw_train['cases']=labels_raw_train["total_cases"]

features_raw_train.head(5)
#%% Lags
frt = features_raw_train.copy()
frt['ln_cases'] = frt.cases.apply(lambda x: np.log(x + 0.0001))
frt = frt.sort_values(['city', 'year', 'weekofyear'])

#%%
frt['week_num'] = [i for i in range(frt.shape[0])]
frt[['week_num','city']].head(10)
frt.loc[frt.city=='sj'][['week_num','city']].head()


#%%
min_recs = frt.groupby('city').agg({'week_num':'min'}).reset_index()
min_recs.rename(columns={'week_num':'min_week_num'}, inplace = True)
frt2 = pd.merge(frt, min_recs, on='city', how='left')
frt2.head(2)
#%%
frt2['week_num'] = frt2['week_num'] - frt2['min_week_num']
frt2.drop(columns={'min_week_num'})
frt = frt2.copy()
del frt2

#%%
frt.groupby('city').agg({'week_num':'min'}).reset_index()
frt.drop(columns=['min_week_num'])

#%%
def lag_variable(
        inputdf
        , varname  = None
        , lagsize  = None
        , groupvar = 'city'
        , ordervar = 'week_num'
        ):
    tdf = inputdf[[groupvar, ordervar, varname]].copy()
    tdf[ordervar] = tdf[ordervar] + lagsize
    varname_new = varname+'_l' + str(lagsize)
    tdf.rename(columns = {varname: varname_new}, inplace = True)
    inputdf_new = pd.merge(inputdf, tdf, on=[groupvar, ordervar], how='left')
    return(inputdf_new) 

#%%
tdf1 = lag_variable(
        frt
        , varname  = 'station_max_temp_c'
        , lagsize  = 2
        , groupvar = 'city'
        , ordervar = 'week_num'
        )
#works
#%%
condition_vars = frt.columns
# drop a few non-lagging ones
condition_vars = [i for i in condition_vars if i not in 
                  ['city','year', 'week_num',
                   'weekofyear','week_start_date']]
#%%
for i in range(1,5):
    print(i)
    
for tvar in condition_vars:
    print(tvar)
    frt2 = lag_variable(frt, varname = tvar, lagsize = i)
    frt = frt2.copy()
#%%
    frt.head(5)
    
    
    
 #%%












































print(df_sj.shape)
print(df_iq.shape)

features_raw_train['cases']= labels_raw_train['total_cases']

g=sns.lmplot(x='ndvi_ne', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='ndvi_nw', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='ndvi_se', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='ndvi_sw', y='cases', data=features_raw_train, markers='o', col='city', hue='city')


features_raw_train['c_lt_10']=features_raw_train['cases'].apply(lambda x: int(x<10))
features_raw_train['c_10_25']=features_raw_train['cases'].apply(lambda x: int(x<=10& x<=25))
features_raw_train['c_gt_25']=features_raw_train['cases'].apply(lambda x: int(x>25))




features_raw_train['cases']= labels_raw_train['total_cases']
g=sns.lmplot(x='precipitation_amt_mm', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='reanalysis_air_temp_k', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='reanalysis_dew_point_temp_k', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='reanalysis_max_air_temp_k', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='reanalysis_min_air_temp_k', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='reanalysis_precip_amt_kg_per_m2', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='reanalysis_specific_humidity_g_per_kg', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='reanalysis_tdtr_k', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='station_avg_temp_c', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='station_diur_temp_rng_c', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='station_max_temp_c', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='station_min_temp_c', y='cases', data=features_raw_train, markers='o', col='city', hue='city')
g=sns.lmplot(x='station_precip_mm', y='cases', data=features_raw_train, markers='o', col='city', hue='city')

#%%
df_sj = features_raw_train[features_raw_train['city'] == 'sj']
df_iq = features_raw_train[features_raw_train['city'] == 'iq']
print(df_sj.shape)
print(df_iq.shape)

sj_corr=df_sj.corr()
iq_corr=df_iq.corr()

sns.heatmap(sj_corr)
plt.title('Correlation Plot of all features in the San Juan Dataset')
plt.show()

sns.heatmap(iq_corr)
plt.title('Correlation Plot of all features in the Iquitos Dataset')
plt.show()

sns.set(font_scale = 1.05)
(abs(sj_corr)
 .cases
 .drop('cases')
 .sort_values()
 .plot
 .barh())


sns.set(font_scale = 1.05)
(abs(iq_corr)
 .cases
 .drop('cases')
 .sort_values()
 .plot
 .barh())

frt= features_raw_train.copy()

frt['ln_cases']=frt.cases.apply(lambda x:np.log(x+0.01)) 

inputdf=ftr.copy()
var= 'station_avg_temp_c'
lag=3


frt=frt.sort_values(['city','year','weekofyear'])
frt['week_num']=[i for i in range(frt.shape[0])]

frt.group_by('city').agg({'week_num',})

def lag_var(inputdf,var,lag):
    