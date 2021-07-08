# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:29:34 2020

@author: aishwary
"""


### Libraries
from os import chdir
chdir('G:/My Drive/2020/Coding/MTA Local')
import pandas as pd
import itertools
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from math import factorial
import copy


### Data Preparation
data = pd.read_csv('attribution data.csv')
df = copy.deepcopy(data)
#df = data.sample(frac=0.1, replace=True, random_state=13)
df = df.sort_values(['cookie', 'time'], ascending=[False, True])
df['visit_order'] = df.groupby('cookie').cumcount() + 1
df_paths = df.groupby('cookie')
df_paths = df_paths['channel'].aggregate(lambda x: x.unique().tolist()).reset_index()
df_last_interaction = df.drop_duplicates('cookie', keep='last')[['cookie', 'conversion']]
df_paths = pd.merge(df_paths, df_last_interaction, how='left', on='cookie')
df_paths['path'] = ''
for i in tqdm(range(len(df_paths))):
    df_paths['path'][i] = '+'.join(df_paths['channel'][i])
df_paths['path'] = df_paths['path'].astype(str)
df2 = df_paths.groupby('path', as_index = False).agg({'conversion':'sum' , 'cookie':'count'})
df2 = df2.drop('cookie', axis = 1)


### Paths - Channels & Coalitions
def unique_paths(list1): 
    list_of_unique_paths = [] 
    for x in list1:
        if x not in list_of_unique_paths:
            list_of_unique_paths.append(x) 
    return list_of_unique_paths

def unique_channels(data_column):
    unique_channels = []
    N = []
    for element in data_column:
        for x in element:
            N.append(x)
    unique_channels = set(N)
    return unique_channels

list_of_all_paths = df_paths['channel']
list_of_unique_paths = unique_paths(list_of_all_paths)
list_of_unique_channels = unique_channels(list_of_all_paths)


### Get Conversions by path
def subsets_permutation(s):
    '''This function returns all the possible subsets of a set of channels.
    input :- s: a set of channels.'''
    if len(s)==1:
        return s
    else:
        sub_channels=[]
        for i in range(1,len(s)+1):
            a = list(map(list,itertools.permutations(s, i)))
            sub_channels.extend(a)
    return list(map('+'.join, sub_channels)) 

### Get Combinations
def subsets_combination(s):
    '''This function returns all the possible subsets of a set of channels.
    input :- s: a set of channels.'''
    if len(s)==1:
        return s
    else:
        sub_channels=[]
        for i in range(1,len(s)+1):
            a = list(map(list,itertools.combinations(s, i)))
            sub_channels.extend(a)
    return list(map('+'.join, sub_channels)) 


def v_function(A,C_values):
    '''This function computes the worth of each coalition.
    inputs:- A : a coalition of channels. - C_values : A dictionnary containing the number 
    of conversions that each subset of channels has yielded.'''
    subsets_of_A = subsets_permutation(A.split("+"))
    worth_of_A = 0
    for subset in subsets_of_A:
        if subset in C_values:
            worth_of_A += C_values[subset]
    return worth_of_A


# Convert conversion dataframe to dictionary
C_values = df2.set_index("path").to_dict()["conversion"]
#For each possible coalition  A, we compute the total number of conversions yielded by every subset of A. 
v_values = {}
for A in subsets_permutation(list_of_unique_channels):
    v_values[A] = v_function(A,C_values)

# check = pd.DataFrame.from_dict(v_values, orient='index')
# check.to_csv('check.csv')


### Shapley Values 
n=len(list_of_unique_channels)
shapley_values = defaultdict(int)

for channel in list_of_unique_channels:
    for A in v_values.keys():
        if channel not in A.split("+"):
            cardinal_A=len(A.split("+"))
            A_with_channel = A.split("+")
            A_with_channel.append(channel)            
            A_with_channel="+".join(sorted(A_with_channel))
            weight = (factorial(cardinal_A)*factorial(n-cardinal_A-1)/factorial(n))
            shapley_values[channel] += (np.max(v_values[A_with_channel])-v_values[A])*weight/factorial(cardinal_A)
    # Add the term corresponding to the empty set
    shapley_values[channel]+= (v_values[channel]/n)

