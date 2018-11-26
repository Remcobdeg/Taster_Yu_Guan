# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:54:09 2018

@author: b8058356
"""

def create_batches(data_x,data_y,range_B, random_start = False):
    dim_data = train_x.shape[1]
    n_classes = train_y.shape[1] 
    
    #determine the batche size in range
    B = np.random.randint(range_B[0],range_B[1],1)[0] #use [0] because the function returns an array and we want a number only
    #the new length of the data
    #T = data_x.shape[0]//B 
    T = data_x.shape[0] #for convinience
    
    #define the sample at which each batch starts
    if random_start == True:
        #randomly chose the first sample for each batch
        start_sample = np.random.randint(low=1,high=T*(1-1/B),size=B) 
    else:
        start_sample = np.array(np.arange(B))*T//B
    #reshape x_train to be separated into batches (batch number in the 3rd dimension)
    data_x_3D = np.zeros((B,T//B,dim_data), dtype=np.float32) #create an empty matrix first
    data_y_3D = np.zeros((B,T//B,n_classes), dtype=np.uint8)
    for batch in range(B):
        data_x_3D[batch,:,:] = data_x[(start_sample[batch]):(start_sample[batch]+T//B),:]
        data_y_3D[batch,:,:] = data_y[(start_sample[batch]):(start_sample[batch]+T//B),:]
            
    print("the data is seperated in {0:d} batches of length {1:d}".format(B,T//B))
    
    #determine coverage
    coverage = np.zeros((B,T//B)) #initialize an empty matrix
    #store for every batch the samples that it covers:
    for batch in range(B): coverage[batch,:] = np.arange((start_sample[batch]),(start_sample[batch]+T//B))
    #make an array of all the unique covered samples and compare it to all possible samples
    coverage = len(np.unique(np.reshape(coverage, (-1))))/T

    return data_x_3D, data_y_3D, coverage


data_x_3D, data_y_3D, coverage = create_batches(train_x,train_y,[128,256], random_start = True)