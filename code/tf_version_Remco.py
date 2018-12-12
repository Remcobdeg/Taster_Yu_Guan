# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:46:03 2018

@author: b8058356 (Remco Benthem de Grave)
"""

#structure of the code:
#A loading libraries
#B Defining parameters ---SELECT THE PARAMETERS HERE----
#C Defining support functions
#   1 reading data from database 
#   2 creating batches of data, added as a 3rd layer to the data
#   3 creating the model structure
#   4 creating model functions
#   5 training the model
#   6 assess the performance of the model (used for both the validation and test set)
#   7 plot the confusion matrix
#   8 core function that call all other functions 



#### A - LOAD LIBRARIES ####

import tensorflow as tf #gerneral deep learning package
import scipy.io #for reading .mat datasets
import numpy as np #general package for useful programming commands
from sklearn.metrics import f1_score #calculates test performance score
from sklearn.metrics import confusion_matrix #performance diagnostic tool
import matplotlib.pyplot as plt #for making a visual graph of the confusion matrix
import itertools #used in plotting the confusion matrix
import os #to check whether a path exists
import pandas as pd #to create dataframes
import shutil #to delete folders
import csv #to print dictionaries
from datetime import datetime #get time stamps
from time import time #calculate time difference
#import pdb #debugging package - use by including 'pdb.set_trace()' in code
#use pdb.set_trace() and call variable by typing "p a_variable" in the console

#### B - DEFINE PARAMETERS ####

def get_parameters():
    
    parameters = {
        'overwrite' : [False,True][1], #whether to overwrite files if storage folder already exists
        'GPU' : [False,True][1], #use GPU for tensorflow model execution
        
        #choose which database to use
        'dataset' : ['Opportunity','Skoda','PAPAM2'][0], 
        
        #Model parameters
        'nodes' : 256,
        'n_layers' : 2,
        
        #Set hyperparameters
        'range_B' : [128, 256], #range of batch size when training
        'range_L' : [16,32], #range of window length when training
        'dropout' : .5,
        'n_epochs' : 1, #epochs per model
        'learning_rate' :0.001,
        
        #Model embedding structure 
        'n_CE_models' : 100, #create n models using CE loss 
        'n_F1_models' : 100, #create n models using F1 loss 
        'embedded_models' : # [best_n_models_CE-loss, best_n_models_F1-loss]
            np.array([[1,0],[0,1],[10,0],[0,10],[20,0],[0,20],[10,10]]),
#             np.array([[1,0],[0,1],[5,0],[0,5],[5,5]]),
       
        #Set validation parameters
        'valid_window' : 5000,
        
        #Set test parameters
        'test_window' : 5000
        }
    
    return parameters


#### 3 SUPPORT FUNCTIONS ####

#### 1 LOAD DATA ####

def load_data(dataset):
	
    if dataset=='Opportunity': 
        
        matfile = '../'+'Ensemble-datasets/OPP/Opp79'+'.mat'
        print(matfile)
        data = scipy.io.loadmat(matfile) #loading the data as dictionary

        X_train = np.transpose(data['trainingData'])
        X_valid = np.transpose(data['valData'])
        X_test = np.transpose(data['testingData'])
        print('normalising... zero mean, unit variance')
        mn_trn = np.mean(X_train, axis=0)
        std_trn = np.std(X_train, axis=0)
        X_train = (X_train - mn_trn)/std_trn
        X_valid = (X_valid - mn_trn)/std_trn
        X_test = (X_test - mn_trn)/std_trn
        print('normalising...X_train, X_valid, X_test... done')
        
        #labels 0-16 instead of 1-17
        y_train = data['trainingLabels'].reshape(-1)-1 
        y_valid = data['valLabels'].reshape(-1)-1
        y_test = data['testingLabels'].reshape(-1)-1

        #reshape label vectors into one-hot dataframes
        y_train = pd.get_dummies( y_train , prefix='labels')
        y_valid = pd.get_dummies( y_valid , prefix='labels' )
        y_test = pd.get_dummies( y_test , prefix='labels' )
        
        #label 17 doesn't occur in the validation set. Add a column of zero's to the dataframe of zeros
        y_valid.insert(17, 'labels_17', 0, allow_duplicates=False)
        print('loading the 79-dim matData successfully . . .')
        
    if dataset=='Skoda':
        
        matfile = '../'+'Ensemble-datasets/Skoda.mat'
        data = scipy.io.loadmat(matfile)
        
        X_train = data['X_train']
        X_valid = data['X_valid']
        X_test = data['X_test']
        y_train = data['y_train'].reshape(-1)
        y_valid = data['y_valid'].reshape(-1)
        y_test = data['y_test'].reshape(-1)
        y_train = pd.get_dummies( y_train , prefix='labels' )
        y_valid = pd.get_dummies( y_valid , prefix='labels' )
        y_test = pd.get_dummies( y_test , prefix='labels' )
        
        print('the Skoda dataset was normalized to zero-mean, unit variance')
        print('loading the 33HZ 60d matData successfully . . .')
	
    if dataset=='PAPAM2':
        matfile = '../'+'Ensemble-datasets/PAMAP2.mat'
        data = scipy.io.loadmat(matfile)
		
        X_train = data['X_train']
        X_valid = data['X_valid']
        X_test = data['X_test']
        y_train = data['y_train'].reshape(-1)
        y_valid = data['y_valid'].reshape(-1)
        y_test = data['y_test'].reshape(-1)
        
        y_train = pd.get_dummies( y_train , prefix='labels' )
        y_valid = pd.get_dummies( y_valid , prefix='labels' )
        y_test = pd.get_dummies( y_test , prefix='labels' )
		
        print('the PAMAP2 dataset was normalized to zero-mean, unit variance')
        print('loading the 33HZ PAMAP2 52d matData successfully . . .')
	
    X_train = X_train.astype(np.float32)
    X_valid = X_valid.astype(np.float32)
    X_test = X_test.astype(np.float32)
   
    y_train = y_train.astype(np.uint8)
    y_valid = y_valid.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
	
    return X_train, X_valid, X_test, y_train, y_valid, y_test

#### 2 CREATING BATCHES OF DATA ####
# creates batches of data and stacks those on top of one another, creating a 3D dataset
# notes: 
#   the new sequence length T_B is determined by T//B
#   the starting point of each batch is randomly chosen from a uniform distribution

def create_batches(data_x,data_y,range_B, seed, random_start = False):
    dim_data = data_x.shape[1]
    n_classes = data_y.shape[1] 
    
    #determine the batche size in range
    seed += 1 #increase seed every time it is used
    np.random.seed(seed) #set seed for reproduction porposes
    B = np.random.randint(range_B[0],range_B[1],1)[0] #use [0] because the function returns an array and we want a number only
    #the new length of the data
    #T = data_x.shape[0]//B 
    T = data_x.shape[0] #for convinience
    
    #define the sample at which each batch starts
    if random_start == True:
        #randomly chose the first sample for each batch
        seed += 1 #increase seed every time it is used
        np.random.seed(seed) #set seed for reproduction porposes
        start_sample = np.random.randint(low=1,high=T*(1-1/B),size=B) 
    else:
        start_sample = np.array(np.arange(B))*T//B
    #reshape x_train to be separated into batches (batch number in the 3rd dimension)
    train_x_3D = np.zeros((B,T//B,dim_data), dtype=np.float32) #create an empty matrix first
    train_y_3D = np.zeros((B,T//B,n_classes), dtype=np.uint8)
    for batch in range(B):
        train_x_3D[batch,:,:] = data_x[(start_sample[batch]):(start_sample[batch]+T//B),:]
        train_y_3D[batch,:,:] = data_y[(start_sample[batch]):(start_sample[batch]+T//B),:]
            
    print("the data is seperated in {0:d} batches of length {1:d} (training sequence)".format(B,T//B))
    
    #determine coverage
    coverage = np.zeros((B,T//B)) #initialize an empty matrix
    #store for every batch the samples that it covers:
    for batch in range(B): coverage[batch,:] = np.arange((start_sample[batch]),(start_sample[batch]+T//B))
    #make an array of all the unique covered samples and compare it to all possible samples
    coverage = len(np.unique(np.reshape(coverage, (-1))))/T
    print("coverage = %.3f" % (coverage))

    return train_x_3D, train_y_3D, coverage, seed


#### 3 CREATING THE MODEL STRUCTURE ####
    
def create_placeholders(dims, classes, n_layers, nodes):
    
    x = tf.placeholder('float', [None,None,dims]) #shape: time steps, batch size, dims
    y = tf.placeholder('float', [None,None,classes])
    states_in = tf.placeholder('float', [n_layers,2,None,nodes]) #2 because LSTM has 2 states to be passed on (both a hidden and cell state)
    keep_prob = tf.placeholder('float', []) #dropout placeholder; single value (value changes for validation and test phase thus needs to be passed as tensor)

    return x, y, states_in, keep_prob

def HAR_model(x, states_in, keep_prob, n_layers, nodes, classes):
    
    
    #reshape the structure of states into a tuple that is recognized by the tf LSTM structure
    states_as_tuple = tuple([tf.nn.rnn_cell.LSTMStateTuple(states_in[layer][0],states_in[layer][1]) for layer in range(n_layers)])
    
    #define a single LSTM layer
    def lstm_cell(nodes,keep_prob):
        return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(nodes),
                                             output_keep_prob=keep_prob,
                                             state_keep_prob=keep_prob)
    
    #combine LSTM layers    
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(nodes,keep_prob) for _ in range(n_layers)])
    
    #apply dynamic unrolling (necessary to allow variation in the sequence length of the input)
    output, state = tf.nn.dynamic_rnn(stacked_lstm, x, initial_state=states_as_tuple)
    
    #push output through a dense layer with linear activation function to prepare it for the loss function
    output = tf.layers.dense(output, units = classes, activation = None)
    
    return output, state

#### 4 CREATING MODEL FUNCTIONS ####

def model_exe_funcs(output,y,losstype,learning_rate):
    
    #define predictions and actual label functions - used later for calculating accuracy and f1-score
    f_prediction = tf.argmax(output,2) #index of max value along the features axis
    f_actual = tf.argmax(y,2) #correct label for each predition  
    
    #get the prediction as a probability distribution (needed for f1-loss and the essembly)
    f_prediction_probs = tf.nn.softmax(output)
    
    #define the cost function    
    if losstype == 'CE':
        f_cost = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(logits = output, onehot_labels = y))
    elif losstype == 'F1':
        #formula 18 in Guan, Y., & Ploetz, T. (2017). Ensembles of Deep LSTM Learners for Activity Recognition using Wearables. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol, 1(11). https://doi.org/10.1145/3090076
        f_cost = 1 - (2*tf.reduce_sum(tf.multiply(f_prediction_probs,y))/(tf.reduce_sum(f_prediction_probs) + tf.reduce_sum(y)))
    
    #define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(f_cost)
    
    init = tf.global_variables_initializer() #initializer
    
    return f_prediction, f_actual, f_prediction_probs, f_cost, optimizer, init

#### 5 TRAINING THE MODEL ####
    
def train_models(train_x, train_y, dropout, #data
                 x, y, states_in, keep_prob, #placeholders
                 state, #from forward propagation
                 f_prediction, f_actual, f_cost, optimizer, init, #modelling functions
                 n_layers, nodes, #for setting initial state
                 range_B, range_L, seed, #for constructing mini-batches
                 n_epochs,GPU):
    
    if GPU: #configuring GPU use for tensorflow models
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config) # is use gpu
    else: sess = tf.Session() #or use only the CPU   
        
    #initialize variables
    sess.run(init)
    
    
    #reshape data into batches starting at different starting points
    train_x_3D, train_y_3D, coverage, seed = create_batches(train_x,train_y,range_B, seed, random_start = True)
    B = train_x_3D.shape[0] #batch size
    L_batch = train_x_3D.shape[1]
    
    #run training loop 
    for epoch in range(n_epochs):
        
        
        #initialize states
        states = np.zeros((n_layers,2,train_x_3D.shape[0],nodes))
        
        #initialize containers
        loss_epoch = 0 #updating the cost after each batch
        L_seq = 0 #length of data currently processed
        i = 0 #counter of the number of windows processed
        actual_epoch = []
        prediction_epoch = []
        
        #update parameters per window length
        while L_seq < L_batch:
            
            #update and set random seed
            seed += 1 #change seed each window
            np.random.seed(seed) #set seed for reproduction porposes

            #define window length 
            L_i = np.random.randint(low=range_L[0],high=range_L[1],size=1)[0] #window
 
            #define minibatches
            minibatch_x = train_x_3D[:,L_seq:(L_seq+L_i),:]
            minibatch_y = train_y_3D[:,L_seq:(L_seq+L_i),:]                
            
            #select data to feed into the training step
            feed_dict = {x: minibatch_x, y: minibatch_y, 
                         states_in: states, keep_prob: 1-dropout}
            
            #train model
            _, loss, states, prediction, actual = sess.run([optimizer, f_cost, state, f_prediction, f_actual], feed_dict = feed_dict)
                    
            #flatten 'prediction' and 'actual' for calculating accuracy and f1
            prediction = np.reshape(np.array(prediction), (-1))
            actual = np.reshape(np.array(actual), (-1))
            
            #calculate the accuracy
            accuracy = np.mean(prediction == actual)
            
            #calculate f1-score: F1 = 2 * (precision * recall) / (precision + recall)
            f1 = f1_score(y_true = actual, y_pred = prediction, average='macro')                
            #macro: Calculate metrics for each label of multi-class labels, and find their unweighted mean.

            L_seq += L_i #update processed sequence length
            processed_sz = B*L_seq #number of samples currently processed

            #print intermediate results to see if the model converges
            i += 1
            if i % 20 == 0:
                print("Epoch %i, after %i windows trainded %i samples, avg. loss = %.3f, accuracy = %.2f, f1-score: %.2f" % (epoch+1, i, processed_sz,loss,accuracy,f1))
#                print("Classes in the true labels:")
#                print(np.unique(actual))
#                print("Classes in the prediction:")
#                print(np.unique(prediction))
                
            #combine mini-batch-wise results to determine epoch-wise results
            actual_epoch.extend(actual)
            prediction_epoch.extend(prediction)
            loss_epoch += loss * B * L_i #'loss' is average loss, we need total loss
            
        #calculate the epoch-wise accuracy
        #actual_epoch = np.reshape(np.array(actual_epoch), (-1))
        #prediction_epoch = np.reshape(np.array(prediction_epoch), (-1))
        accuracy = np.mean(np.array(prediction_epoch) == np.array(actual_epoch))
        
        #calculate epoch-wise f1-score: F1 = 2 * (precision * recall) / (precision + recall)
        f1 = f1_score(y_true = actual_epoch, y_pred = prediction_epoch, average='macro')                
 
        print("Result epoch %i of %i: \n average loss: %.3f, accuracy: %.2f, f1-score: %.2f" 
              % (epoch+1, n_epochs, loss_epoch/processed_sz, accuracy, f1))
        
    return sess, accuracy, f1, seed


#### 6 ASSESS MODEL PERFORMANCE (VALIDATE/TEST THE MODEL) ####
    
def eval_models(sess, 
                    data_x, data_y, #data
                    x, y, states_in, keep_prob, #placeholders
                    state, #from forward propagation
                    f_prediction, f_actual, f_prediction_probs, f_cost, init, #modelling functions
                    n_layers, nodes, #for setting initial state
                    data_window): 
    
    #initialize states to zero
    states = np.zeros((n_layers,2,1,nodes)) #2 states in each unrolled LSTM cell, batch size = 1
    
    #initialize containers
    prediction = [] #container for predictions in every window
    actual = []  #container for actual labels in every window
    prediction_probs = np.zeros((data_y.shape)) #container for probabilities in every window
    loss = 0 #container for the validation/test loss
    
    #the model expects the data to be 3D (with batch size as first dim) - our batch size = 1 for the test and validation
    data_x = np.reshape(data_x, (1, data_x.shape[0], data_x.shape[1]))
    data_y = np.reshape(data_y, (1, data_y.shape[0], data_y.shape[1]))
    
    #loop over the number of times the window fits in the data (rounded upwards, because we want to use all the data)
    for i in range(np.ceil(data_x.shape[1]/data_window).astype(np.int)):
        
        #define the sample indices included in the window; for the last window, a shorter window needs to be chosen
        window = range(data_window*i, 
                            np.min([data_window*(i+1),data_x.shape[1]]))

        feed_dict = {x: data_x[:,window,:], y: data_y[:,window,:], 
                     states_in: states, keep_prob: 1}
        
        #run the model to calculate loss, predictions, actual lables - i refers to the window number
        loss_i, states, prediction_i, actual_i, prediction_probs_i = sess.run(
                [f_cost, state, f_prediction, f_actual, f_prediction_probs], feed_dict = feed_dict)
        
        #flatten and store 'prediction' and 'actual' for calculating accuracy and f1
        prediction.extend(np.reshape(np.array(prediction_i), (-1)))
        actual.extend(np.reshape(np.array(actual_i), (-1)))
        prediction_probs[window,:] = prediction_probs_i
        
        #to calculate the average loss over the validation after processing the 
        #whole batch, we need to devide the total loss by the whole sample size
        #loss.append(loss_i*len(window))
        
        loss += loss_i * len(window) #translate average loss to sum loss (because last window has different size)
        
    #calculate the average validation/test loss
    loss = loss/data_x.shape[1]
    
    #calculate the accuracy
    accuracy = np.mean(np.array(prediction) == np.array(actual))
    
    #calculate f1-score: F1 = 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(y_true = np.array(actual), y_pred = np.array(prediction), average='macro')                
    #macro: Calculate metrics for each label of multi-class labels, and find their unweighted mean.
    
#    print("Classes in the true labels:")
#    print(np.unique(actual))
#    print("Classes in the prediction:")
#    print(np.unique(prediction))
    
    print("Result model evaluation: \n loss: %.3f, accuracy: %.2f, f1-score: %.2f \n" % 
          (loss, accuracy, f1))
    
    return accuracy, f1, prediction_probs


#### 7 PLOT THE CONFUSION MATRIX ####
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,path='./'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)



#### 8 CORE FUNCTION ####         

def make_model(parameters):
    
    #time stamp for calculating preperation time
    start_time = time()
    
    #read the parameters
    overwrite = parameters['overwrite']
    GPU = parameters['GPU']
    dataset = parameters['dataset'] 
    nodes = parameters['nodes']
    n_layers = parameters['n_layers']
    range_B = parameters['range_B'] 
    range_L = parameters['range_L'] 
    dropout = parameters['dropout']
    n_epochs = parameters['n_epochs'] 
    learning_rate = parameters['learning_rate']
    n_CE_models = parameters['n_CE_models'] 
    n_F1_models = parameters['n_F1_models'] 
    embedded_models = parameters['embedded_models']
    valid_window = parameters['valid_window']
    test_window = parameters['test_window']
    
    
    #check for existance of a previous unfinished process
    continue_prev_process = 'N' #initialize
    if os.path.isfile('process_state.npy'):
        process_state = np.load('process_state.npy').item()
        
        if process_state['testing'] != 'completed':
            print('\n The is an unfinhed process:')
            print(process_state)
            print('\n Continue from unfinished process?')
            continue_prev_process = input('Y/N: \n')
    
    
    if continue_prev_process == 'N':
        #initializing a new process
        
        model = 0 #counter used when training the models 
        seed = 0 #initialize seed - used to make runs comparible
        
        #print a starting message
        print("\n \n START \n")
        print("creating " + str(n_CE_models) + " models with CE-loss and " +
              str(n_F1_models) + " models with F1-loss. \n \n")
        
        #name the folder after the dataset used and the time
        folder = dataset + datetime.now().strftime('%Y-%m-%d_%H%M') 
        
        #check if the folder already exist and adapt name if necessary
        while os.path.isdir("./model/" + folder):
            if overwrite:
                shutil.rmtree("./model/" + folder)
            else: folder = folder + "dub"        
        folder = folder + "/" 
        os.makedirs("./model/" + folder) #make the folder
        os.makedirs("./results/" + folder) #make the folder
        
        print("\n models will be stored in " + os.getcwd() + './model/' + folder)
        print("\n results will be stored in " + os.getcwd() + './results/' + folder)
        
        #store the parameter values
        with open('./results/' + folder + 'parameters.csv', 'w') as f:  # 'w' for 'write'
            w = csv.DictWriter(f, parameters.keys())
            w.writeheader()
            w.writerow(parameters)
        
        #create a dictionary for storing the results of each single model
        pre_test_results = {'model' : [], 'losstype' : [] ,'acc_train' : [], 'f1_train' : [],
                        'acc_valid' : [], 'f1_valid' : [], 'train_time' : []}
        
    else:
        #continue from previous process
        model = process_state['last_model']
        seed = process_state['seed']
        folder = process_state['folder']
        pre_test_results = pd.read_csv('./results/' + folder + 
                                       'results_train_valid.csv', index_col = 0).to_dict(orient='list')

        print('\n Continuing previous process \n')
        
    
    ## LOAD DATA
    
    #load the database through the function 'loadingDB()' from dataset.py    
    train_x, valid_x, test_x, train_y, valid_y, test_y = load_data(dataset)
    
    #the y-variables have been loaded as databases, which causes some manipulation challenges
    #thus we will reshape those to numpy arrays
    train_y = np.array(train_y)
    valid_y = np.array(valid_y)
    test_y = np.array(test_y)
    
#    ###### TEMP STEPS TO SPEED UP TRAINING: REDUCE DATA SIZE 10x #######
#    train_x = train_x[:round(train_x.shape[0]/10),:]
#    train_y = train_y[:round(train_y.shape[0]/10),:]
#    range_B = [16,32]
#    #####################################################################
  
    dims = train_x.shape[1] #number of dimensions/features in the data
    classes = train_y.shape[1] #number of classes in the labels
    
    
    ## PREPARE TENSORFLOW GRAPH
    
    #first reset the computational graph (tensorflow struction) - 
    #this prevents problems with running code a second time without closing python
    tf.reset_default_graph()
    
    #create tensorflow variable placeholders
    x, y, states_in, keep_prob = create_placeholders(dims, classes, n_layers, nodes)
    
    #Create model/Forward propagation: Build the forward propagation in the tensorflow graph
    output, state = HAR_model(x, states_in, keep_prob, n_layers, nodes, classes) 

    #create a function to save weights and biases from training
    #we want to store many models, so set max_keep to a high number
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10000)
        
    #determine preparation time
    prep_time = (time()-start_time)/60 #in minutes
    
    
    ## TRAIN AND VALIDATE THE SINGLE MODELS    
    
    #per model that we create we train, assess the performance of the
    # single model on the validation set it and store results
    while model < (n_CE_models + n_F1_models):
        
        #stamp time for calculating time difference
        start_time = time()
        
        if model < n_CE_models: losstype = 'CE'    
        else: losstype = 'F1' 
        
        #define model name based on loss-type used and n-th model created of that loss-type
        model_name = 'model_' + str(model+1) + '_' + losstype
        
        print("\n Training "+model_name)
        
        #define model execution parameters (only f_cost depends on the loss-type)
        f_prediction, f_actual, f_prediction_probs, f_cost, optimizer, init = model_exe_funcs(output,y,losstype,learning_rate)
    
        #train the model
        sess, accuracy_train, f1_train, seed = train_models(train_x, train_y, dropout, #data
                         x, y, states_in, keep_prob, #placeholders
                         state, #from forward propagation
                         f_prediction, f_actual, f_cost, optimizer, init, #modelling functions
                         n_layers, nodes, #for setting initial state
                         range_B, range_L, seed, #for constructing mini-batches
                         n_epochs, GPU)
        
        #store results
        pre_test_results['model'].append(model_name)
        pre_test_results['losstype'].append(losstype)
        pre_test_results['acc_train'].append(accuracy_train)
        pre_test_results['f1_train'].append(f1_train)
    
        #save the model
        saver.save(sess, './model/' + folder + model_name)
        
        #perform validation
        accuracy_valid, f1_valid, _ = eval_models(sess, 
                    valid_x, valid_y, #data
                    x, y, states_in, keep_prob, #placeholders
                    state, #from forward propagation
                    f_prediction, f_actual, f_prediction_probs, f_cost, init, #modelling functions
                    n_layers, nodes, #for setting initial state
                    valid_window)
            
        pre_test_results['acc_valid'].append(accuracy_valid)
        pre_test_results['f1_valid'].append(f1_valid)
        
        model = model + 1 #update model counter
        
        #register training time
        pre_test_results['train_time'].append((time()-start_time)/60)
        
        #storing of results in dataframe format to csv file
        pd.DataFrame(pre_test_results).to_csv('./results/' + folder + 'results_train_valid.csv')
                
        #saving current status in both essemble folder and main folder
        np.save('process_state.npy', {'folder' : folder, 'last_model' : model, 
                                      'out_of' : n_CE_models + n_F1_models, 
                                      'seed' : seed, 'testing' : 'not started'})
        
       
        sess.close() 
    
    
    ## CREATING EMBEDDED STRUCTURES AND ASSESS PERFORMANCE ON TEST SET
            
    #we select the 'best' models on the f1 performance on the validation set (others scores are for information purpose only)
    
    #create a dictionary for storing the results of each essembly
    test_results = {'model' : [], 'accuracy' : [], 'f1' : [], 'test_time' : []}
    
    #sort the validation results based on the performance on f1
    pre_test_results = pd.DataFrame(pre_test_results).sort_values(by=['f1_valid'], ascending=False) 
        
    #the next steps are performed per embedded model structure created
    for essembly in embedded_models:
        
        #start a time couter
        start_time = time()
        
        #get the model references of the n-best models
        best_CE = np.where(pre_test_results.loc[:]['losstype'] == 'CE')[0][:essembly[0]]
        best_F1 = np.where(pre_test_results.loc[:]['losstype'] == 'F1')[0][:essembly[1]]
        best_models = np.concatenate((best_CE, best_F1))
                
        #initialize a matrix to store the probabilities of the predictions from the models that build the embedded model
        prediction_probs_all_models = np.zeros((len(best_models),test_y.shape[0],test_y.shape[1])) 
        
        if GPU: #configuring GPU use for tensorflow models
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config) # is use gpu
        else: sess = tf.Session() #or use only the CPU   
        
        
        #get predictions (probability distribution) for each single model in the embbeded structure
        
        i = 0 #counter parrellel to the 'model' index, because 'model' is not from 0:n_models
        
        for model in best_models:
            
            #find the name of the corresponding model
            model_name = pre_test_results.loc[model]['model']
            
            #load the weights and biases from the corresponding session
            saver.restore(sess, './model/' + folder + model_name)
            
            losstype = pre_test_results.loc[model]['losstype']
            
            #define model execution parameters (only f_cost depends on the loss-type)
            f_prediction, f_actual, f_prediction_probs, f_cost, _, init = model_exe_funcs(output,y,losstype,learning_rate)
            
            #get the predictions
            _,_,prediction_probs = eval_models(sess, 
                            test_x, test_y, #data
                            x, y, states_in, keep_prob, #placeholders
                            state, #from forward propagation
                            f_prediction, f_actual, f_prediction_probs, f_cost, init, #modelling functions
                            n_layers, nodes, #for setting initial state
                            test_window)
            
            prediction_probs_all_models[i,:,:] = prediction_probs
            
            i = i + 1 #update counter
            
        sess.close()
        
        #get the combined predictions from all the models (note that 
        #it doesn't matter whether we get the average or sum)            
        prediction_probs_all_models = np.sum(prediction_probs_all_models, axis = 0)
        
        #get the class of the probability
        prediction = np.argmax(prediction_probs_all_models, axis = -1)
        #and the actual class
        actual = np.argmax(test_y, axis = -1)
        accuracy = np.mean(prediction == actual)
        f1 = f1_score(y_true = np.array(actual), y_pred = np.array(prediction), average='macro')
        
        
        #make and save the confusion matrix
        essembly_name = str(essembly[0]) + '_CE_' + str(essembly[1]) + '_F1_essembly'
        labels = ["%i" % label for label in np.arange(classes)]
        cm = confusion_matrix(prediction, actual)
        path = './results/' + folder + 'confusion_' + essembly_name
        plot_confusion_matrix(cm, labels, path = path + '.jpg') #save as picture
        #np.savetxt(path + '.csv', cm, fmt = '%d', delimiter=',') #as csv file
        pd.DataFrame(cm, index = ['A'+label for label in labels], columns = ['P'+label for label in labels]).to_csv(path + '.csv') #as dataframe

        print("Embedded model has accuracy of %.2f and f1-score of %.2f" % (accuracy, f1))    
        
        #add results to dictionary
        test_results['model'].append(essembly_name)
        test_results['accuracy'].append(accuracy)
        test_results['f1'].append(f1)
        test_results['test_time'].append((time()-start_time)/60)
        
    #storing of test results in dataframe format to csv file
    pd.DataFrame(test_results).to_csv('./results/' + folder + 'test_results.csv')
    
    #store file with information about the time taken
    process_time = {'prep_time' : prep_time, 'avg_train_time' : np.mean(pre_test_results['train_time']),
                    'avg_test_time' : np.mean(test_results['test_time'])}
    with open('./results/' + folder + 'process_time.csv', 'w') as f:  # 'w' for 'write'
        w = csv.DictWriter(f, process_time.keys())
        w.writeheader()
        w.writerow(process_time)
    
    #print an overview
    print(parameters)
    print(pre_test_results)
    print(test_results)
    
    #update the process state to being finished
    np.save('process_state.npy', {'folder' : folder, 'last_model' : '', 
                          'out_of' : n_CE_models + n_F1_models, 
                          'seed' : seed, 'testing' : 'completed'})

    print("\n COMPLETE")        
    print("\n Data stored in " + os.getcwd() + './results/' + folder)



 
     
#### RUN CODE ####
make_model(get_parameters())
