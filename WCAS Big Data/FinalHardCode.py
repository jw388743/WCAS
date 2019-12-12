#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:17:35 2019

@author: james
"""

#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


projFld = "/Users/james/Desktop/ADEC7430"
codeFld = os.path.join(projFld, "Code")
fnsFld = os.path.join(codeFld,"_Functions")
outputFld = os.path.join(projFld, "Output")
rawDataFld = os.path.join(projFld, "RawData")
savedDataFld = os.path.join(projFld, "SavedData")
os.path.exists(outputFld)
dataFld="/Users/james/Desktop/data"
#%%
player_raw=os.path.join(dataFld, 'player.csv')
player=pd.read_csv(player_raw)
stats_raw_file=os.path.join(dataFld, 'season.csv')
stats_raw=pd.read_csv(stats_raw_file)
#%%
pd.set_option('display.width',100)
pd.set_option('max_colwidth',50)
pd.set_option('display.max_columns',100)
#%%
print(player.dtypes)
print(stats_raw.dtypes)
print(player.shape)
print(stats_raw.shape)

#%%
player=player.sort_values('year_start').reset_index()
#%%
player.drop(columns=['index', 'year_start', 'year_end', 'birth_date',
                     'college'],inplace=True)
print(player.shape)

#%%
player['Player']=player['name']
player.drop(columns=['name'],inplace=True)
#%%
print(stats_raw[stats_raw.Player=='Larry Bird*'].count())
print(player[player.Player=='Larry Bird*'].count())
print(stats_raw[stats_raw.Player=='Curly Armstrong'].count())
print(player[player.Player=='Curly Armstrong'].count())
#%%
print(player.shape)
print(player['Player'].isin(stats_raw['Player']).value_counts())
print(stats_raw['Player'].isin(player['Player']).value_counts())
#%%
stats_clean = stats_raw.drop(['blanl', 'blank2', 'Tm'], axis=1)
#%%
cond = player['Player'].isin(stats_raw['Player']) == False
player.drop(player[cond].index, inplace = True)
print(player.shape)
#%%
print(stats_clean.shape)
df_stats= pd.merge(stats_clean, player[['Player', 'height', 'weight']], how='right', 
                          left_on='Player', right_on='Player', right_index=False,sort=False)
print(df_stats.shape)
print(player['Player'].isin(df_stats['Player']).value_counts())
print(df_stats['Player'].isin(player['Player']).value_counts())
#%%
print(df_stats[df_stats.Player=='Larry Bird*'].count())
print(player[player.Player=='Larry Bird*'].count())
print(df_stats[df_stats.Player=='Curly Armstrong'].count())
print(player[player.Player=='Curly Armstrong'].count())

#%%
df_stats
#%%
pd.isnull(df_stats).astype(int).aggregate(sum).to_dict() 
#%%
print(df_stats['Pos'].value_counts())
plt.figure(figsize=(15,5))
sns.countplot(x="Pos", data=df_stats, palette="Set3")

#%%
hybrid_indx= df_stats[(df_stats['Pos']!='PG') & (df_stats['Pos']!='SG')
                          & (df_stats['Pos']!='SF') & (df_stats['Pos']!='PF')
                          &(df_stats['Pos']!='C')].index
df_stats=df_stats.drop(hybrid_indx, axis=0)
print(df_stats.shape)
#%%
stats= df_stats[df_stats.Year>=1980]
stats=stats.reset_index()
print(stats.shape)

#%%
def encode_era(df_):
    df=df_.copy()
    eras=['80s','90s','2000s','2010s']
    cut_bins=[1979,1989,1999,2009,2019]
    df['era']=pd.cut(df['Year'], bins=cut_bins, labels=eras)
    return(df)
stats=encode_era(stats)
stats[stats.Year==2009]
#%%

#%%
# split Train/Test
from sklearn.model_selection import train_test_split as skl_traintest_split
X_stats = stats.copy().drop(columns={'index','Unnamed: 0','Player'})

X_stats_train, X_stats_test = skl_traintest_split(X_stats, test_size=0.30, random_state=2019)
#%%
print(X_stats_train.shape)
print(X_stats_test.shape)
print(X_stats_train['era'].value_counts())
print(X_stats_test['era'].value_counts())
print(X_stats_train['Pos'].value_counts())
print(X_stats_test['Pos'].value_counts())

#%%
pd.isnull(X_stats_train).astype(int).aggregate(sum).to_dict() 
#%%
pd.isnull(X_stats_test).astype(int).aggregate(sum).to_dict() 
#%%
import pickle as pkl
#%%
X_stats_test_file= os.path.join(savedDataFld, "X_stats_test.pkl")
X_stats_test.to_pickle(X_stats_test_file)
X_stats_train_file= os.path.join(savedDataFld, "X_stats_train.pkl")
X_stats_train.to_pickle(X_stats_train_file)
del X_stats_test
#%%
pd.isnull(X_stats_train).astype(int).aggregate(sum).to_dict() 
#%% 5 min per game players
print(X_stats_train.shape)
X_stats_train=X_stats_train[X_stats_train['MP']>=400]
print(X_stats_train.shape)
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, col='Pos', hue='Pos')
g.map(sns.lineplot,'Year', "3P%",err_style=None)
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, col='Pos', hue='Pos')
g.map(sns.lineplot,'Year', "USG%",err_style=None)
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, col='Pos', hue='Pos')
g.map(sns.lineplot,'Year', "3PA",err_style=None)
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, col='Pos', hue='Pos')
g.map(sns.lineplot,'Year', "FGA",err_style=None)
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, col='Pos', hue='Pos')
g.map(sns.lineplot,'Year', "PER",err_style=None)
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, col='Pos', hue='Pos')
g.map(sns.lineplot,'Year', "PTS",err_style=None)
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, col='Pos', hue='Pos')
g.map(sns.lineplot,'Year', "FTA",err_style=None)
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, col='Pos', hue='Pos')
g.map(sns.lineplot,'Year', "USG%",err_style=None)
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, row='era', col='Pos', hue='Pos')
g.map(sns.lineplot,'PTS', "3PA",err_style=None)
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, row='era', col='Pos', hue='Pos')
g.map(sns.scatterplot,'PTS', "3P%")
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, row='era', col='Pos', hue='Pos')
g.map(sns.scatterplot,'3PA', "3P%")
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, row='era', col='Pos', hue='Pos')
g.map(sns.scatterplot,'PTS', "3PA")
#%%
plt.figure(figsize=(16, 6))
g = sns.FacetGrid(X_stats_train, row='era', col='Pos', hue='Pos')
g.map(sns.scatterplot,'PTS', "eFG%")
#%%
#%%
cond_totals =['PER', 'OWS', 'DWS', 'WS', 'OBPM', 'DBPM', 'BPM', 'VORP', 'FG', 
          'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA',
          'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
cond_percents= X_stats_train['TS%','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%',
                             'USG%','OWS','DWS','WS','WS/48','FG%','3P%','2P%','eFG%']
#%%
intresting_conds=['PTS','PER','FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA',
                  'FG%','3P%','2P%','eFG%']
#%%
train_corr= X_stats_train.corr()
sns.heatmap(train_corr)
plt.title('Correlation Heat Plot')
plt.show()
#%%
X_stats_percent=X_stats_train.drop(columns=cond_totals, axis=1).copy()
X_stats_tot=X_stats_train.drop(columns=cond_percents, axis=1).copy()
#%%
percent_corr= X_stats_percent.corr()
sns.heatmap(percent_corr)
plt.title('Correlation Heat Plot')
plt.show()
#%%
tot_corr= X_stats_tot.corr()
sns.heatmap(tot_corr)
plt.title('Correlation Heat Plot')
plt.show()
#%%
X_stats_intrest=X_stats_train.drop(columns=intresting_conds, axis=1).copy()
#%%
intrest_corr= X_stats_intrest.corr()
sns.heatmap(intrest_corr)
plt.title('Correlation Heat Plot')
plt.show()



#%%
tlist = X_stats_train.groupby(['Pos','era']).agg('median').reset_index().to_dict('records')
#%%
tdict=tdict = {'C_80s': tlist[0], 'C_90s': tlist[1],'C_2000s': tlist[2],
               'C_2010s': tlist[3], 'PF_80s': tlist[4],'PF_90s': tlist[5],
               'PF_2000s': tlist[6],'PF_2010s': tlist[7],
               'PG_80s': tlist[8],'PG_90s': tlist[9],
               'PG_2000s': tlist[10],'PG_2010s': tlist[11],
               'SF_80s': tlist[12],'SF_90s': tlist[13],
               'SF_2000s': tlist[14],'SF_2010s': tlist[15],
               'SG_80s': tlist[16],'SG_90s': tlist[17],
               'SG_2000s': tlist[18],'SG_2010s': tlist[19]}
#%%
for tvar in X_stats_train.columns:
    print(tvar)
    if tvar in ['Year']:
        pass
    else:
        tfilt_C_80s=pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='80s') & (X_stats_train['Pos']=='C')
        print('tfilt_C_80s=',sum(tfilt_C_80s))
        X_stats_train.loc[tfilt_C_80s, tvar] = tdict['C_80s'][tvar]
        tfilt_C_90s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='90s') & (X_stats_train['Pos']=='C')
        print('tfilt_C_90s=',sum(tfilt_C_90s))
        X_stats_train.loc[tfilt_C_90s, tvar] = tdict['C_90s'][tvar]
        tfilt_C_2000s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='2000s') & (X_stats_train['Pos']=='C')
        print('tfilt_C_2000s=',sum(tfilt_C_2000s))
        X_stats_train.loc[tfilt_C_2000s, tvar] = tdict['C_2000s'][tvar]
        tfilt_C_2010s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='2010s') & (X_stats_train['Pos']=='C')
        print('tfilt_C_2010s=',sum(tfilt_C_2010s))
        X_stats_train.loc[tfilt_C_2010s, tvar] = tdict['C_2010s'][tvar]
        tfilt_PF_80s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='80s') & (X_stats_train['Pos']=='PF')
        print('tfilt_PF_80s=',sum(tfilt_PF_80s))
        X_stats_train.loc[tfilt_PF_80s, tvar] = tdict['PF_80s'][tvar]
        tfilt_PF_90s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='90s') & (X_stats_train['Pos']=='PF')
        print('tfilt_PF_90s=',sum(tfilt_PF_90s))
        X_stats_train.loc[tfilt_PF_90s, tvar] = tdict['PF_90s'][tvar]
        tfilt_PF_2000s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='2000s') & (X_stats_train['Pos']=='PF')
        print('tfilt_PF_2000s=',sum(tfilt_PF_2000s))
        X_stats_train.loc[tfilt_PF_2000s, tvar] = tdict['PF_2000s'][tvar]
        tfilt_PF_2010s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='2010s') & (X_stats_train['Pos']=='PF')
        print('tfilt_PF_2010s=',sum(tfilt_PF_2010s))
        X_stats_train.loc[tfilt_PF_2010s, tvar] = tdict['PF_2010s'][tvar]
        tfilt_PG_80s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='80s') & (X_stats_train['Pos']=='PG')
        print('tfilt_PG_80s=',sum(tfilt_PG_80s))
        X_stats_train.loc[tfilt_PG_80s, tvar] = tdict['PG_80s'][tvar]
        tfilt_PG_90s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='90s') & (X_stats_train['Pos']=='PG')
        print('tfilt_PG_90s=',sum(tfilt_PG_90s))
        X_stats_train.loc[tfilt_PG_90s, tvar] = tdict['PG_90s'][tvar]
        tfilt_PG_2000s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='2000s') & (X_stats_train['Pos']=='PG')
        print('tfilt_PG_2000s=',sum(tfilt_PG_2000s))
        X_stats_train.loc[tfilt_PG_2000s, tvar] = tdict['PG_2000s'][tvar]
        tfilt_PG_2010s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='2010s') & (X_stats_train['Pos']=='PG')
        print('tfilt_PG_2010s=',sum(tfilt_PG_2010s))
        X_stats_train.loc[tfilt_PG_2010s, tvar] = tdict['PG_2010s'][tvar]
        tfilt_SF_80s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='80s') & (X_stats_train['Pos']=='SF')
        print('tfilt_SF_80s=',sum(tfilt_SF_80s))
        X_stats_train.loc[tfilt_SF_80s, tvar] = tdict['SF_80s'][tvar]
        tfilt_SF_90s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='90s') & (X_stats_train['Pos']=='SF')
        print('tfilt_SF_90s=',sum(tfilt_SF_90s))
        X_stats_train.loc[tfilt_SF_90s, tvar] = tdict['SF_90s'][tvar]
        tfilt_SF_2000s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='2000s') & (X_stats_train['Pos']=='SF')
        print('tfilt_SF_2000s=',sum(tfilt_SF_2000s))
        X_stats_train.loc[tfilt_SF_2000s, tvar] = tdict['SF_2000s'][tvar]
        tfilt_SF_2010s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='2010s') & (X_stats_train['Pos']=='SF')
        print('tfilt_SF_2010s=',sum(tfilt_SF_2010s))
        X_stats_train.loc[tfilt_SF_2010s, tvar] = tdict['SF_2010s'][tvar]
        tfilt_SG_80s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='80s') & (X_stats_train['Pos']=='SG')
        print('tfilt_SG_80s=',sum(tfilt_SG_80s))
        X_stats_train.loc[tfilt_SG_80s, tvar] = tdict['SG_80s'][tvar]
        tfilt_SG_90s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='90s') & (X_stats_train['Pos']=='SG')
        print('tfilt_SG_90s=',sum(tfilt_SG_90s))
        X_stats_train.loc[tfilt_SG_90s, tvar] = tdict['SG_90s'][tvar]
        tfilt_SG_2000s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='2000s') & (X_stats_train['Pos']=='SG')
        print('tfilt_SG_2000s=',sum(tfilt_SG_2000s))
        X_stats_train.loc[tfilt_SG_2000s, tvar] = tdict['SG_2000s'][tvar]
        tfilt_SG_2010s =pd.isnull(X_stats_train[tvar]) & (X_stats_train['era']=='2010s') & (X_stats_train['Pos']=='SG')
        print('tfilt_SG_2010s=',sum(tfilt_SG_2010s))
        X_stats_train.loc[tfilt_SG_2010s, tvar] = tdict['SG_2010s'][tvar]
        
pd.isnull(X_stats_train).astype(int).aggregate(sum).to_dict() 
#%%
cond_totals = ['PER', 'OWS', 'DWS', 'WS', 'OBPM', 'DBPM', 'BPM', 'VORP', 'FG', 
          'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA',
          'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
cond_percents= ['TS%','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%',
           'USG%','OWS','DWS','WS','WS/48','FG%','3P%','2P%','eFG%']
#%%
def encode_era_dummy(df_):
    df=df_.copy()
    df['era_80s'] = 0
    df['era_90s'] = 0
    df['era_2000s'] = 0
    df['era_2010s'] = 0
    df['era_post_2010s'] = 0
    filter_ = df.era=="80s"
    df.loc[filter_, 'era_80s'] = 1
    filter_ = df.era=="90s"
    df.loc[filter_, 'era_90s'] = 1
    filter_ = df.era=="2000s"
    df.loc[filter_, 'era_2000s'] = 1
    filter_ = df.era=="2010s"
    df.loc[filter_, 'era_2010s'] = 1
    filter_ = df.era.isin(['80s','90s','2000s','2010s'])
    filter_.value_counts(dropna=False)
    df.loc[~filter_, 'era_post_2010s'] = 1
    return(df)
#%%
def confusionMatrixInfo(p,a, labels = None):
    from sklearn import metrics as skm
    from sklearn.metrics import confusion_matrix as skm_conf_mat
    import sys
    #
#    p = pd.Series([1,1,1,0,0,0,0,0,0,0])
#    a = pd.Series([1,0,0,1,1,1,0,0,0,0])
#    labels = [1,0]
#
#    x = skm.confusion_matrix(a,p,labels=labels)
    if 'sklearn' not in sys.modules:
        import sklearn
    x = skm_conf_mat(a,p, labels = labels)
    tp = x[0,0]
    tn = x[1,1]
    fp = x[1,0]
    fn = x[0,1]
    # tp, fp, fn, tn # test order
    
    tsensitivity = tp/(tp+fn)
    tspecificity = tn/(tn + fp)
    # no information rate?
    tnir = (tp + fn)/x.sum()
    tnir = max(tnir, 1-tnir)
    # accuracy
    taccuracy = (tp + tn)/x.sum()
    
    res = {'confusionMatrix':x,
           'accuracy': taccuracy,
           'no information rate': tnir,
           'sensitivity': tsensitivity,
           'specificity': tspecificity
           }
    return(res)
#%%https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
#%%























#%% Nueral Network
#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

cclass Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

























