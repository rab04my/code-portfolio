#################################################################################################
# Experimenting with different methods for disaggregating (or splitting) signals from a single 
# energy signal (i.e. at the meter).
#
### NOTES
# Research concepts from https://www.sciencedirect.com/science/article/pii/S0378778818304146
#  Applying NMF decomposition requires to split the time series PotenciaPF (whole house) into windows in order to
# generate the columns of the input matrix V. We set the size of the windows so that the obtained columns
# were daily observations of the total demand of the hospital. With this arrangement, the resulting coefficient
# vector Hi represents the daily contributions of basis consumption Wi throughout the year. In this process,
# we discarded those days which presented missing records, because NMF would learn the gaps in the data as
# components, instead of other relevant patterns in the network.
#  The results of NMF are strongly dependent on the length of the windows. Thus, if we choose a weekly or annually
# instance of the load profile, NMF will decrease resolution in the obtained patterns, revealing weekly or annual
# patterns, losing resolution in daily events. Therefore, the size of the window should be chosen according to
# the desired analysis. In our work, we will focus on interpretable events in daily scale because they can be
# associated with daily behaviors that can be easily corrected, so each window will be a daily instance of the
# load profile.
#  Consider that the results obtained by the basic NMF are sparse enough, so that we do not integrate
# measurements of sparsity in the cost function
#################################################################################################

import os
import sys
import requests
import numpy as np
from scipy import *
import pandas as pd
import calendar
import dateutil
import datetime
from plotnine import *
import seaborn as sns
from mizani.breaks import date_breaks
from mizani.formatters import date_format
import subprocess
import warnings
warnings.filterwarnings("ignore") #category=DeprecationWarning

np.random.seed(42)

## read meter data
path = '/Users/rab04my/Documents/ev_detection/disaggregation-project/data/'
file = 'prem_550414157_July2018toJuly2019_kw_data_bryan.csv'

df = pd.read_csv(path+file)
df.columns

df = df.fillna(0)
df['kw'] = df['watt']/1000
df['read_time'] = pd.to_datetime(df['read_time'], format='%Y-%m-%d %H:%M:%S')
df = df[df['kw']<=50].drop('lag', axis=1)
timestamp_orig = df.read_time

df_nmf = df.copy()

## filter data by date (if needed)
df.set_index('read_time', inplace=True)
df = df.loc['2019-07-25 00:00:00':'2019-07-25 23:59:59'] # 6/13/2019 = Thursday
df = df.resample('1T').sum().reset_index()
df.shape
df.head()

df.to_csv('/Users/rab04my/Documents/ev_detection/disaggregation-project/fhmm/src/prem_disagg_data.csv',
          columns=['read_time', 'kw'], date_format='%Y-%m-%d %H:%M:%S', index = None, header=True)

## set number of components/groups manually
groups = 8

#########################################
### (1) HMM
from hmmlearn import hmm

## copy and use original data frame
HMM = df.copy()

# get 'discrete' kW values
kw = HMM.kw.values
kw = [[kw[i]] for i in range(0, len(kw), 1)]

## try both hmm.GMMHMM() and hmm.GaussianHMM(); can also decode(), predict_proba(),
#hmm_model = hmm.GaussianHMM(n_components=groups, covariance_type="full", n_iter=100)
hmm_model = hmm.GMMHMM(n_components=groups, covariance_type="full", verbose=False, n_iter=100)
hmm_model.fit(kw)
#val, pred = hmm_model.sample(100) # sample from HMM
## can train a model and predict on current kW values (model fit is SLOW)
HMM['label'] = hmm_model.predict(kw)
HMM['label'] = HMM.label.astype("category", ordered=False, inplace=True)
HMM.label.value_counts()

# combine any overlapping time/grouping values (#z.index.get_level_values(0))
HMM = HMM.groupby(['prem_num','read_time','label',]).sum().dropna().reset_index()
HMM.set_index(HMM['read_time'], inplace=True)
HMM.index = pd.to_datetime(HMM.index)

## change names to be consistent throughout
HMM.columns = ['prem_num', 'read_time', 'label', 'watt', 'kw']

## frequency counts/totals
#HMM['label'].value_counts()
#HMM.groupby('label')['kw'].sum()
#HMM.head()

## HMM plots
## time series coloured by predicted group (appliance)
m1 = (ggplot(HMM, aes(x="read_time", y="kw")) +
    #geom_col(aes(fill="label", group="read_time"), alpha=0.8) +
    geom_line(aes(color="label", group="read_time")) +
    #geom_point(aes(color="label", group="read_time"), alpha=0.4) +
    scale_x_datetime(breaks=date_breaks('1 hour'), labels=date_format('%a %H:%M')) +
    theme_bw() +
    theme(axis_text_x=element_text(rotation=90, hjust=0.5))
)
m1
#ggsave(plot=m1, filename=path+'ami_minute_barplot_bryan_25july2019_EST.png', dpi=800)


#######
# from IOHMM import UnSupervisedIOHMM
# from IOHMM import OLS, DiscreteMNL, CrossEntropyMNL

#######
# import bhmm
#
# obs = [np.array([0, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int),
#                np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0], dtype=int)
#                ]
# len(obs)
# kw2 = np.asarray(kw)
# b = bhmm.estimate_hmm(kw2, 6, lag=1, output='gaussian')
#######

#########################################
### Factorial HMM
### NOTE THAT SOMETIMES IT FAILS AND NEED TO DO IT MANUALLY
### Also seems to have issues running data less than 1440 minutes per day (i.e. missing)
### https://github.com/sjjsy/sor-nilm (in R)
### > Rscript r_fhmm.R -b -i prem727755_3June2019_data.csv -o test.csv -v -n 100
rscript = '/usr/local/bin/Rscript'
script_file = '/Users/rab04my/Documents/ev_detection/disaggregation-project/fhmm/src/r_fhmm2.R'
input_file = '/Users/rab04my/Documents/ev_detection/disaggregation-project/fhmm/src/prem_disagg_data.csv'
output_file = '/Users/rab04my/Documents/ev_detection/disaggregation-project/fhmm/src/fhmm_disagg_output.csv'
cmd = 'exec '+rscript+' '+script_file+' -b -i '+input_file+' -o '+output_file+' -v -a 5 -n 100 -p 5 --seed 42'
print(cmd)

## need to see why this sometimes loops forever???
# res = subprocess.Popen(cmd, stdout=None, stderr=subprocess.PIPE, shell=True)
# if res.wait() != 0:
#     error = res.communicate()
#     print(error)
# res.kill()

FHMM = pd.read_csv(output_file)
#FHMM.head()

mFHMM = pd.melt(FHMM, id_vars=['Timestamp'], var_name='label')
mFHMM.columns = ['read_time', 'label', 'kw']
mFHMM.set_index('read_time', inplace=True)
mFHMM.index = pd.to_datetime(mFHMM.index)
mFHMM.label.unique()

# (ggplot(mFHMM, aes(x='mFHMM.index', y='kw')) +
#  geom_col(aes(fill='label')) +
#  scale_x_date(date_breaks="1 hour", labels=date_format("%a %H:%M")) +
#  labs(x='Hour of the Day', y='Usage (kW)', title='', color='Label') +
#  theme_bw() +
#  theme(axis_text_x=element_text(rotation=90, hjust=0.5))
# )

#########################################
### Create dataset using 1-minute kW data with each column indicating a day of usage (no index)
### To be used with NMF, UMAP, Matrix Representation, Clustering, etc.
## copy and use original data frame
#X1 = df.copy()
X1 = df_nmf.copy()

X1 = X1.reset_index()
X1['date'] = pd.to_datetime(X1.read_time).dt.strftime('%Y-%m-%d')
X1 = X1.set_index(X1.date)
X1 = X1.drop(['prem_num','watt','date'], axis=1) #'read_time',
X1.columns
X1.head()

Xlist = []
Xlist = [group[1] for group in X1.groupby(X1.index)]
date_labels = X1.index.array.unique()

day_df = pd.DataFrame()
for i in range(0, len(Xlist), 1):
    day_df = pd.concat([day_df, Xlist[i].reset_index().drop('date', axis=1)['kw']], axis=1, ignore_index=True)

#day_df.describe()
day_df = day_df.fillna(0)

day_df.set_index(X1.read_time.dt.strftime('%H:%M:%S')[:1440], inplace=True)
day_df.columns = date_labels
day_df = day_df.fillna(0)
day_df.describe()

#########################################
### NMF
### Use this due to the positive nature of electric power demand and because the
### imposed constraints on the resulting components may improve their interpretability
### Similar to PCA, the basis vectors (components) are in order of relevance?
### Maybe use clustering off of the reduced space matrix H?
###
### or use similarity to clustering e.g.
### 1) Perform NMF based on Lee and Seung [2] on V and get W and H
### 2) Apply  cosine  similarity  to  measure  distance  between  the documents di and
###    extracted features/vectors of W.
### 3) Assign  di  to  wx  if  the  angle  between  di  and  wx    is smallest. This is
###    equivalent to k-means algorithm with a single turn.
## get estimate NMF table (reconstruction of original) - not sure this is needed at this point??
#mw = np.matrix(W)
#mh = np.matrix(H)
#mx = np.matmul(mw, mh)
#tmp_mat = pd.DataFrame(mx).transpose()
#tmp_mat['type'] = "nmf_est"
#tmp_mat.head()

# the product (Wα %*% Hα,j) between one basis consumption α (cell in H defined by row α and column j) and
# its coefficient corresponding to a specific observation j (in the W matrix) is unique and it has units of
# measurement (kW in the case of total electric power demand).
# [Where α is a value/element (row #) and j is the column/component]

# Note to get back to kW, need to multiply the specific value of W corresponding with the associated (i,j)
# [row and column for a specific cell value].
#(150, 6) ~ W
#(6, 1440) ~ H

from sklearn.decomposition import NMF
import nimfa #another package (not used) to do NMF (http://nimfa.biolab.si/index.html)

## original 1-D vector (do not use)
#X1D = z['kw'].values.reshape(1, -1)

nmf = NMF(n_components=groups, max_iter=200, solver="mu")
W = nmf.fit_transform(day_df.transpose())
H = nmf.components_
W.shape
H.shape
groups

val = []
out = []
out2 = []
for c in range(0,W.shape[1],1):
    for i in range(0,W.shape[0],1):
        w1 = W[i,c]
        for j in range(0,H.shape[1],1):
            h1 = H[c,j]
            val.append(w1*h1)
        out.append(val)
        val = []
    out2.append(out)
    out = []

len(out2[0][0])
len(out2[0])
len(out2)

def nmf_df_prep(dat, group):
    z = pd.DataFrame(dat[group]).transpose()
    z.columns = date_labels
    z.set_index(X1.read_time.dt.strftime('%H:%M:%S')[:1440], inplace=True)
    z['label'] = 'Group ' + str(group)

    z = pd.melt(z.reset_index(), id_vars=['read_time', 'label'], var_name='date')
    z['date'] = z['date'].map(str)
    z['time'] = z['read_time'].map(str)
    z['read_time'] = z['date'] + ' ' + z['time']
    z['read_time'] = pd.to_datetime(z['read_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    z.drop(['time','date'], axis=1, inplace=True)
    z.set_index('read_time', inplace=True)
    z.index = pd.to_datetime(z.index)
    z.columns = ['label', 'kw']
    return(z)

NMF = nmf_df_prep(dat=out2, group=0)
for i in range(1,groups,1):
    tmp = nmf_df_prep(dat=out2, group=i)
    NMF = pd.concat([NMF, nmf_df_prep(dat=out2, group=i)])

# (ggplot(NMF, aes(x='NMF.index', y='kw')) +
#  geom_col(aes(fill='label')) +
#  facet_wrap("~label", scales="free_y") +
#  scale_x_datetime(breaks=date_breaks('5 day'), labels=date_format('%a, %b %d')) +
#  theme_bw() +
#  theme(axis_text_x=element_text(rotation=90, hjust=0.5))
# )

## hourly
NMF2 = NMF.groupby('label').resample('H').sum().reset_index()
NMF2.set_index('read_time', inplace=True)

# (ggplot(NMF2, aes(x='NMF2.index', y='kw')) +
#  geom_col(aes(fill='label')) +
#  scale_x_datetime(breaks=date_breaks('1 day'), labels=date_format('%a, %b %d')) +
#  theme_bw() +
#  theme(axis_text_x=element_text(rotation=90, hjust=0.5))
# )


from sklearn.cluster import KMeans
import hdbscan

clust = hdbscan.HDBSCAN(algorithm='best', alpha=0.5, approx_min_span_tree=True, gen_min_span_tree=False,
                        leaf_size=10, metric='euclidean', min_cluster_size=4, min_samples=25, p=None)

clust.fit(H.transpose())
clust.labels_.max()
clust.labels_
clust.probabilities_
clust.labels_.shape
test = ([clust.labels_]*9)
flattened_list = [y for x in test for y in x]
len(flattened_list)

### factorial_hmm package (https://github.com/regevs/factorial_hmm/wiki/Constructing-a-model)
sys.path.append('/Users/rab04my/Documents/ev_detection/disaggregation-project/')
from factorial_hmm import *

random_state = np.random.RandomState(0)

## Gaussian (continuous) FHMM modeling




# start discrete FHMM example (from site)
F = MyFullDiscreteFactorialHMM(n_steps=100)
Z, X = F.Simulate()

tmp = HMM.kw.values
tmp = np.array([tmp.tolist()])

# use tmp or X (simulated data)
R = F.EM(tmp, likelihood_precision=0.1, n_iterations=1000, verbose=True, print_every=1, random_seed=None)

R.initial_hidden_state_tensor

R.transition_matrices_tensor[0]

R.obs_given_hidden[0]

alphas, betas, gammas, scaling_constants, log_likelihood = F.EStep(observed_states=X)
Y = F.DrawFromPosterior(observed_states=X, alphas=alphas, betas=betas, scaling_constants=scaling_constants, start=0, end=None, random_seed=None)
most_likely, back_pointers, lls = F.Viterbi(observed_states=X)
# end discrete example


#########################################
### COMBINE results
hmm1 = HMM.copy()
fhmm1 = mFHMM.copy()
nmf1 = NMF.copy()
nmf1 = nmf1.loc['2019-07-25 00:00:00':'2019-07-25 23:59:59']
#nmf2 = NMF.copy()
#nmf2 = nmf2.loc['2019-07-25 00:00:00':'2019-07-25 23:59:59']
#nmf2['label'] = flattened_list

hmm1['method'] = 'HMM'
hmm1['label'] = hmm1['label'].map(str)
fhmm1['method'] = 'FHMM'
nmf1['method'] = 'NMF'
#nmf2['method'] = 'NMF_Clustering'

results = hmm1[['read_time', 'kw', 'label', 'method']].append(fhmm1, ignore_index=False)
results = results.append(nmf1, ignore_index=False)
#results = results.append(nmf2, ignore_index=False)
#results.set_index('read_time', inplace=True)
results.drop('read_time', axis=1, inplace=True)
results['day'] = results.index.day.map(str)
#results['month'] = results.index.month
#results.index
tab = results.groupby(['method', 'label'])['kw'].sum()
tab.groupby('method').sum()

r2 = results.groupby(['method','label','day']).resample('H').sum().reset_index()
#r2.set_index('read_time', inplace=True)

# (ggplot(r2, aes(x='read_time', y='kw', group='label')) +
#  geom_line(aes(color='label'), alpha=0.5) +
#  scale_x_datetime(breaks=date_breaks('2 hour'), labels=date_format('%a %m/%d %H:%M')) +
#  facet_wrap("~method") +
#  labs(x="", y="Usage (kWh)", title="kWh totals (sum) per method") +
#  theme_bw() +
#  theme(axis_text_x=element_text(rotation=90, hjust=0.5))
# )

mytitle = "Bryan's Household July 25, 2019\nStacked by Hours (EST)"

# (ggplot(r2, aes(x="read_time", y="kw", group='label')) +
# #geom_area(aes(fill="sense_name", group="sense_name"), position="identity", alpha=0.2) +
# geom_col(aes(fill="label")) +
# scale_x_datetime(breaks=date_breaks('1 hour'), labels=date_format('%a, %b %d @ %H:%M')) +
# labs(x="", y="Total Usage (kW)", fill="Appliance Group",
#      title=mytitle) +
# facet_wrap("~method+day") +
# theme_bw() +
# theme(axis_text_x=element_text(rotation=90, hjust=0.5))
# )

# p1 = (ggplot(r2[r2.method=='FHMM'], aes(x='read_time', y='kw', group='label')) +
#  #geom_line(alpha=0.5) +
#  geom_col(aes(fill="label")) +
#  scale_x_datetime(breaks=date_breaks('1 hour'), labels=date_format('%a, %b %d, %H:%M')) +
#  facet_wrap("~method+day") +
#  labs(x="", y="Total Usage (kW)", fill="Appliance Group",
#       title=mytitle) +
#  theme_bw() +
#  theme(axis_text_x=element_text(rotation=90, hjust=0.5))
# )

# p2 = (ggplot(r2[r2.method=='HMM'], aes(x='read_time', y='kw', group='label')) +
#  #geom_line(alpha=0.5) +
#  geom_col(aes(fill="label")) +
#  scale_x_datetime(breaks=date_breaks('1 hour'), labels=date_format('%a, %b %d, %H:%M')) +
#  facet_wrap("~method+day") +
#  labs(x="", y="Total Usage (kW)", fill="Appliance Group",
#       title=mytitle) +
#  theme_bw() +
#  theme(axis_text_x=element_text(rotation=90, hjust=0.5))
# )

# p3 = (ggplot(r2[r2.method=='NMF'], aes(x='read_time', y='kw', group='label')) +
#  #geom_line(alpha=0.5) +
#  geom_col(aes(fill="label")) +
#  scale_x_datetime(breaks=date_breaks('1 hour'), labels=date_format('%a, %b %d, %H:%M')) +
#  facet_wrap("~method+day") +
#  labs(x="", y="Total Usage (kW)", fill="Appliance Group",
#       title=mytitle) +
#  theme_bw() +
#  theme(axis_text_x=element_text(rotation=90, hjust=0.5))
# )

# p1
# p2
# p3

# ggsave(plot=p1, filename=path+'fhmm_usage_disagg_by_day_bryan_25july2019_EST.png', dpi=500)
# ggsave(plot=p2, filename=path+'hmm_usage_disagg_by_day_bryan_25july2019_EST.png', dpi=500)
# ggsave(plot=p3, filename=path+'nmf_usage_disagg_by_day_bryan_25july2019_EST.png', dpi=500)

## export results
results.to_csv(path+'prem_???_disagg_results_bryan_25june2019_EST.csv', columns=['method','label','kw'])
os.system("say 'beep'")


#########################################
### umap
### https://umap-learn.readthedocs.io/en/latest/parameters.html
### Similar to NMF... applying UMAP decomposition requires to split the time series data
### into windows in order to generate the columns of the input matrix V. We set the size of
### the windows so that the obtained columns were daily observations of the total demand of
### the hospital. With this arrangement, the resulting coefficient vector Hi represents the
### daily contributions of basis consumption Wi throughout the year.
#########################################
import umap

## use the 'day_df' variable
reducer = umap.UMAP(random_state=42, n_components=2, min_dist=0.1,
                    n_neighbors=15, metric='euclidean')

embedding = reducer.fit_transform(day_df)
embedding.shape

#plt.scatter(u[:,0], u[:,1], c=data)

#########################################
### matrix representation
### REVISIT this with multiple column formatted energy data
#########################################

#########################################
### autoencoder
### Use latent feature layer as the disaggregated appliances?
### Or reconstruct with final nodes equal to # of appliances?
#########################################
import h2o

os.environ['NO_PROXY'] = 'localhost' #'stackoverflow.com' # ‘localhost’
h2o.init(nthreads=-1)
#h2o.no_progress()

## either use 'day_df' or '' object 'df' with single kw column
