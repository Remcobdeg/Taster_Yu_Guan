# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:46:03 2018

@author: b8058356 (Remco Benthem de Grave)
"""

#structure of the code:
#A loading libraries
#B defining functions
#   1 reading data from database - data from various persons are concatenated so
#     we end up with a 2D dataset of nFeatures*samples
#   2 creating batches of data, added as a 3rd layer to the data
#   3 creating the model
#   4 training the model
#   5 assess the performance of the model on a validation or test set
#   6 function that loops through step 2-5 to create n models and selects the m best
#     models (based either on the accuracy or f1 criterion) and runs the chosen
#     essembly on the test set 
#C defining parameters and run the code



#### LOAD LIBRARIES ####
import tensorflow as tf #gerneral deep learning package
from dataset import loadingDB #dataset.py in same folder - shapes train, valid and test datasets
import numpy as np #general package for useful programming commands
from sklearn.metrics import f1_score #calculates test performance score
from sklearn.metrics import confusion_matrix #performance diagnostic tool
import pdb #debugging package - use by including 'pdb.set_trace()' in code

#### 1 LOAD DATA ####

#external function loadingDB in file dataset.py
#data from various persons are concatenated so we have a 2D dataset of nFeatures*samples

#### 2 CREATING BATCHES OF DATA ####
# creates batches of data and stacks those on top of one another, creating a 3D dataset
# notes: 
#   the new sequence length T_B is determined by T//B
#   the starting point of each batch is randomly chosen from a uniform distribution

def create_batches(data_x,data_y,range_B, seed, random_start = False):
    dim_data = data_x.shape[1]
    n_classes = data_y.shape[1] 
    
    #determine the batche size in range
    np.random.seed(seed) #set seed for reproduction porposes
    B = np.random.randint(range_B[0],range_B[1],1)[0] #use [0] because the function returns an array and we want a number only
    #the new length of the data
    #T = data_x.shape[0]//B 
    T = data_x.shape[0] #for convinience
    
    #define the sample at which each batch starts
    if random_start == True:
        #randomly chose the first sample for each batch
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

    return train_x_3D, train_y_3D, coverage


#### 3 CREATING THE MODEL ####
    
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


#### 4 TRAINING THE MODEL ####
    
def train_models(train_x, train_y, dropout, n_layers, nodes, range_B, n_epochs, seed, losstype = 'logloss'):
    
    #first reset the computational graph (tensorflow struction)
    tf.reset_default_graph()

    dims = train_x.shape[1]
    classes = train_y.shape[1]
    
    #create placeholders
    x, y, states_in, keep_prob = create_placeholders(dims, classes, n_layers, nodes)
    
    #Forward propagation: Build the forward propagation in the tensorflow graph
    logits, state = HAR_model(x, states_in, keep_prob, n_layers, 256, classes) 
    
    #define predictions and actual label functions - used later for calculating accuracy and f1-score
    f_prediction = tf.argmax(logits,2) #index of max value along the features axis
    f_actual = tf.argmax(y,2) #correct label for each predition    
    
    #temp position
    probs = tf.nn.softmax(logits) 
    #probs = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis = -1) #softmax to calculate the probability distribution of the classes

    
    #define the cost function    
    if losstype == 'logloss':
        f_cost = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(logits = logits, onehot_labels = y))
    elif losstype == 'f1loss':
        #formula 18 in Guan, Y., & Ploetz, T. (2017). Ensembles of Deep LSTM Learners for Activity Recognition using Wearables. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol, 1(11). https://doi.org/10.1145/3090076
        print(probs)
        f_cost = 1 - (2*tf.reduce_sum(tf.multiply(probs,y))/(tf.reduce_sum(probs) + tf.reduce_sum(y)))
    
    #define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(f_cost)
    
    init = tf.global_variables_initializer() #initializer
    
    with tf.Session() as sess:
        
        #initialize variables
        sess.run(init)
        
        #run training loop 
        for epoch in range(n_epochs):
            
            #reshape data into batches starting at different starting points
            train_x_3D, train_y_3D, coverage = create_batches(train_x,train_y,range_B, seed, random_start = True)
            B = train_x_3D.shape[0] #batch size
            L_batch = train_x_3D.shape[1]
            
            #initialize states
            states = np.zeros((n_layers,2,train_x_3D.shape[0],nodes))
            
            #initialize containers
            sum_loss = 0 #updating the cost after each batch
            L_seq = 0 #length of data currently processed
            i = 0 #counter of the number of windows processed
            
            #update parameters per window length
            while L_seq < L_batch:
                
                #update and set random seed
                seed = seed + 1 #change seed each window
                np.random.seed(seed) #set seed for reproduction porposes

                #define window length 
                L_i = np.random.randint(low=16,high=32,size=1)[0] #window
                #steps = train_x_3D.shape[1]//L_seq
                #print(L_seq, steps, L_seq.dtype, steps.dtype)
                #print("the training batch is processed in %3i steps of %3i samples (window length)" % (L_seq,steps))
                
                #define minibatches
                minibatch_x = train_x_3D[:,L_seq:(L_seq+L_i),:]
                minibatch_y = train_y_3D[:,L_seq:(L_seq+L_i),:]                
                
                #select data to feed into the training step
                feed_dict = {x: minibatch_x, y: minibatch_y, 
                             states_in: states, keep_prob: 1-dropout}
                
                #train model
                _, loss, states, prediction, actual, probs_out,y_out = sess.run([optimizer, f_cost, state, f_prediction, f_actual, probs,y], feed_dict = feed_dict)
                
                print("sum y, sum logits, y*logits")
                print(np.sum(y_out))
                print(np.sum(probs_out))
                print(np.sum(probs_out*y_out))
                
                
                #flatten 'prediction' and 'actual' for calculating accuracy and f1
                prediction = np.reshape(prediction, (-1))
                actual = np.reshape(actual, (-1))
                
                #calculate the accuracy
                accuracy = np.mean(prediction == actual)
                
                #calculate f1-score: F1 = 2 * (precision * recall) / (precision + recall)
                f1 = f1_score(y_true = actual, y_pred = prediction, average='macro')                
                #macro: Calculate metrics for each label of multi-class labels, and find their unweighted mean.
                
                #calculate batch wise cost
                sum_loss += loss

                L_seq += L_i #update processed sequence length
                processed_sz = B*L_seq #number of samples currently processed

                i += 1
                if i % 20 == 0:
                    print("Epoch %i, after %i windows trainded %i samples, loss = %.3f, accuracy = %.2f, f1-score: %.2f" % (epoch, i, processed_sz,sum_loss,accuracy,f1))
 
            print("Epoch-wise loss = %.3f" % (sum_loss))


#### 4 TRAINING THE MODEL ####
    
def validate_models(sess, valid_x, valid_y, x, y, states_in, keep_prob, valid_window, 
                    n_layers, nodes, f_cost,f_prediction, f_actual):
    #dropout = 0
    
    #initialize states to zero
    states = np.zeros((n_layers,2,1,nodes)) #2 states in each unrolled LSTM cell, batch size = 1
    
    #initialize containers
    prediction = [] #container for predictions in every window
    actual = []  #container for actual labels in every window
    loss = [] #container for the validation/test loss
    
    for i in np.ceil(valid_x.shape[0]/valid_window):
        
        #define the sample indices included in the window; for the last window, a shorter window needs to be chosen
        window = range(valid_window*i, 
                            np.min(valid_window*(i+1),valid_x.shape[0]))

        feed_dict = {x: valid_x[:,window,:], y: valid_y[:,window,:], 
                     states_in: states, keep_prob: 1}
        
        #run the model to calculate loss, predictions, actual lables - i refers to the window number
        loss_i,prediction_i, actual_i = sess.run([f_cost,f_prediction, f_actual], feed_dict = feed_dict)
        
        #flatten 'prediction' and 'actual' for calculating accuracy and f1
        loss.extend(loss_i)
        prediction.extend(np.reshape(prediction_i, (-1)))
        actual.extend(np.reshape(actual_i, (-1)))
        
    #calculate the average validation/test loss
    loss = np.mean(loss)
    
    #calculate the accuracy
    accuracy = np.mean(prediction == actual)
    
    #calculate f1-score: F1 = 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(y_true = actual, y_pred = prediction, average='macro')                
    #macro: Calculate metrics for each label of multi-class labels, and find their unweighted mean.
    
    print("validation of %s: \n loss: %.3f, accuracy: %.2f, f1-score: %.2f" % 
          (loss, accuracy, f1))
         

#### PICK BEST X-MODELS FROM VALIDATION SET ####

#### TEST MODEL PERFORMANCE ON TEST SET ####
        
#### C DEFINING PARAMETERS AND RUN THE CODE ####

seed = 0 #initialize seed

#choose which database to use
DB = 79 #79 is opp, 52 is pam, 60 is skoda

#Model parameters
nodes = 256
n_layers = 2

#Set hyperparameters
range_B = [128, 256] #range of batch length
dropout = .5
n_epochs = 2
losstype = 'logloss'

#Set validation parameters
valid_window = 5000


#### RUN CODE ####


#load the database through the function 'loadingDB()' from dataset.py    
#train_x, valid_x, test_x, train_y, valid_y, test_y = loadingDB('../', DB)

#the y-variables have been loaded as databases, which causes some manipulation challenges
#thus we will reshape those to numpy arrays
train_y = np.array(train_y)
valid_y = np.array(valid_y)
test_y = np.array(test_y)



train_models(train_x, train_y, dropout, n_layers, nodes, range_B, n_epochs, seed, losstype = 'logloss')


      