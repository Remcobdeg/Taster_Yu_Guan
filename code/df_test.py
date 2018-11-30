# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:47:46 2018

@author: b8058356
"""

pre_test_results = {'model' : [], 'losstype' : [] ,'acc_train' : [], 'f1_train' : [],
                    'acc_valid' : [], 'f1_valid' : []}

for model in range(5):

    losstype = 'CE'
    model_name = losstype + '_model_' + str(model+1)
    accuracy_train, f1_train = .5, .4
    accuracy_valid, f1_valid = .3, .2
    
    pre_test_results['model'].append(model_name)
    pre_test_results['losstype'].append(losstype)
    pre_test_results['acc_train'].append(accuracy_train)
    pre_test_results['f1_train'].append(f1_train)
    
    pre_test_results['acc_valid'].append(accuracy_valid)
    pre_test_results['f1_valid'].append(f1_valid)

print(pre_test_results)

import pandas as pd

df = pd.DataFrame(pre_test_results)

print(df)

df.to_csv(path_or_buf = 'df.csv')
pd.DataFrame(pre_test_results).to_csv(path_or_buf = 'df2.csv')

read_data = pd.read_csv('df.csv', index_col = 0)

read_data.loc[:]['f1_train'] = read_data.loc[:]['f1_train'] + [3, 4, 1, 2, 5]
read_data.loc[:]['f1_valid'] = read_data.loc[:]['f1_valid'] + [3, 4, 1, 2, 5]

read_data = read_data.sort_values(by=['f1_valid'])

read_data