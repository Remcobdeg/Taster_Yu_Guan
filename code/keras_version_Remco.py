# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:46:03 2018

@author: b8058356 (Remco Benthem de Grave)
"""

#### LOAD LIBRARIES ####
import tensorflow as tf #gerneral deep learning package
from tensorflow.contrib import rnn #recurrent networks
from keras.models import Sequential #keras model structure
from keras.layers import LSTM, Dense,Dropout #keras model features
from keras.utils import np_utils
from dataset import loadingDB #dataset.py in same folder - shapes train, valid and test datasets
import numpy as np #general package for useful programming commands
from sklearn.metrics import f1_score #calculates test performance score
from sklearn.metrics import confusion_matrix #performance diagnostic tool
import pdb #debugging package - use by including 'pdb.set_trace()' in code

#### LOAD DATA ####

setnum = 1 #1 is opp, 2 is pam, 3 is skoda

if setnum == 1:
    DB = 79 #reference to the data set to load below
if setnum == 2:
	DB = 52
if setnum == 3:
    DB = 60

#load the database through the function 'loadingDB()' from dataset.py    
train_x, valid_x, test_x, train_y, valid_y, test_y = loadingDB('../', DB)

#csv = open(str(DB)+'.csv','a') #to store performance results later on


#### FUNCTION THAT DETERMINES THAT DATA SET USED IN EACH ITERATION ####

def create_batches(data_x,data_y,range_split, random_start = False):
    dim_data = train_x.shape[1]
    n_classes = train_y.shape[1] 
    
    #determine number of batches in range (min_n_batches:max_n_batches)
    n_batches = np.random.randint(range_split[0],range_split[1],1)[0] #use [0] because the function returns an array and we want a number only
    #the length of each batch
    l_batches = data_x.shape[0]//n_batches
    
    #define the sample at which each batch starts
    if random_start == True:
        #randomly chose the first sample for each batch
        start_sample = np.random.randint(low=1,high=data_x.shape[0]-l_batches,size=n_batches) 
    else:
        start_sample = np.array(range(n_batches))*l_batches
    #reshape x_train to be separated into batches (batch number in the 3rd dimension)
    data_x_3D = np.zeros((n_batches,l_batches,dim_data), dtype=np.float32) #create an empty matrix first
    data_y_3D = np.zeros((n_batches,l_batches,n_classes), dtype=np.uint8)
    for batch in range(n_batches):
        data_x_3D[batch,:,:] = data_x[start_sample[batch]:start_sample[batch]+l_batches,:]
        data_y_3D[batch,:,:] = data_y[start_sample[batch]:start_sample[batch]+l_batches,:]
            
    print("the data is seperated in {0:d} batches of length {1:d}".format(n_batches,l_batches))

    return data_x_3D, data_y_3D


#### TRAIN AND VALIDATE THE MODEL ####
    
def train_models(n_epochs,losstype = 'logloss'):

    #note that in contrast to the conventional appraoch of running multiple
    #epochs in which the model is improved each time based on the optimized
    #parameters from the previous epoch, the approach here is different:
    #In each epoch, a new model is created. All created models are tested on
    #a validation set and the best (or an essembly of the best ones) is 
    #chosen that will subsequently be used on the test set. 
    #Because of this, below a model is created, trained and validated in each epoch 
    
    #initialize a container to store the various model that are created
    all_models = [] #creates an empty list
    #initialize a container for validation results of each model
    valid_results = []
    
    #parameters that remain stable between the models
    lstm_nodes = 256
    layers = 2
    dropoutrate = .5
    dim_data = train_x.shape[1]
    n_classes = train_y.shape[1] 
    range_n_batches = [128, 256] #min and max number of batches trained in an epoch
    range_l_window = [32, 64] #min and max window (also 'mini-batch') size, defining the batch size
    #...by which the parameters are updated
    
    #for each epoch create, train and validate a new model
    for i in range(n_epochs):
        
        #create batches for bagging procedure as described in the paper
        #shape is: number of batches, length of batches, dimensions
        print("Creating batches:")
        train_x_3D, train_y_3D = create_batches(train_x,np.array(train_y),range_n_batches, random_start = True)
    
        #run the model for each batch seperately
        for j in range(train_x_3D.shape[0]): 
            
            #splitting data in mini-batches of random length that defines the windows on which parameter optimization is done
            #shape is: number of batches, length of batches, dimensions
            print("Creating mini-batches / windows:")
            train_x_mbs, train_y_mbs = create_batches(train_x_3D[j,:,:],train_y_3D[j,:,:],range_l_window, random_start = False)
        
            #create model
            model = Sequential()
            model.add(LSTM(lstm_nodes, input_shape=(train_x_mbs.shape[1], dim_data), 
                           return_sequences=True))
            model.add(Dropout(0.5))
            model.add(LSTM(lstm_nodes, 
                           return_sequences=False))
            model.add(Dropout(0.5))
            model.add(Dense(n_classes, activation='softmax'))
            model.summary()
            
            #compile model 
            model.compile(optimizer='adam', loss='categorical_crossentropy', 
                  metrics=['accuracy'])
            
            #load weights from previous batch (unless this is the first batch of the model)
            if j > 0:
                model.load_weights('./checkpoints/my_checkpoint')
                
            #train model
            history = model.fit(train_x_mbs, train_y_mbs, epochs=1, 
                            batch_size=train_x_mbs.shape[0], shuffle=True, verbose=1)
            #note that we create a new model each epoch, so 'epochs' = 1
            
            # Save the weights to use as a starting point for the next batch of the model
            model.save_weights('./checkpoints/my_checkpoint')
         
        #store the model after batch-wise updating is finished
        all_models[i] = model 
        
        #validate the model and store the result of the validation
        _,valid_results[i] = model.evaluate(valid_x, valid_y) #returns loss and accuracy; store the latter
    
    return all_models, valid_results

#### DEFINE EMBEDDING STRUCTURE AND CALL TRAINING FUNCTION ####

n_models = 100 #number of models that is created in total

lossfunctions = ['logloss','f1','both'][2] #change index to chose type of lossfunction used 
n_embedding = 10 #number of models amongst which aggregation is performed. Note:...
#... if 'both' is chosen as lossfunction, than this number will automatically double

if lossfunctions == 'both':
    #run the training for half of the number of models using logloss 
    all_models1, valid_results1 = train_models(losstype='logloss',n_epochs=np.ceil(n_models/2))
    #get the indices of the n-best models, with n the number of embedded models used
    best_models = np.argsort(valid_results1)[-n_embedding:-1]
    #get the n-best models
    final_model = all_models1[best_models]
    
    #run the training for half of the number of models using f1-loss 
    all_models2, valid_results2 = train_models(losstype='f1',n_epochs=np.ceil(n_models/2))
    #get the indices of the n-best models, with n the number of embedded models used
    best_models = np.argsort(valid_results2)[-n_embedding:-1]
    #get the n-best models
    final_model = [final model, all_models2[best_models]]
    
else: 
    #run the training
    all_models, valid_results = train_models(losstype=losstype,n_epochs=n_models)
    #get the indices of the n-best models, with n the number of embedded models used
    best_models = np.argsort(valid_results)[-n_embedding:-1]
    #get the n-best models
    final_model = all_models[best_models]


#### RUN TEST SET AND REPORT RESULTS ####





#### TRAIN MODEL ####

#### PICK BEST X-MODELS FROM VALIDATION SET ####

#### TEST MODEL PERFORMANCE ON TEST SET ####
        
        
 #### OLD AND UNUSED CODE PARTS - LEFT HERE FOR POSSIBLE RE-USE LATER ####       
        
#    #first define a location to safe, for each model (per epoch) the model weights.
#    #this is necessary, because of the limitations in Keras, we are not able to 
#    #vary the batch sizes during the model training. Instead what the code will do is
#    #recreate the model for each training batch (adapting for the varying batch size)
#    #save the optimized weights and use these again as imput for training on the next batch
#    checkpoint_path = "training_{0:d}/cp.ckpt".format(i) #a folder per epoch
#    checkpoint_dir = os.path.dirname(checkpoint_path)
#    
#    # Create checkpoint callback
#    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
#                                                     save_weights_only=True,
#                                                     verbose=1)
 
 
 
#        train_x_mbs = train_x_3D[j,:,:]
#        train_x_mbs = np.reshape(train_x_mbs, (train_x_mbs.shape[0],1,train_x_mbs.shape[1]))
#        train_y_mbs = train_y_3D[j,:,:]
#        train_y_mbs = np.reshape(train_y_mbs, (train_y_mbs.shape[0],1,train_y_mbs.shape[1]))
#
#        
#        #create model
#        model = Sequential()
#        model.add(LSTM(lstm_nodes, input_shape=(train_x_mbs.shape[1], dim_data), 
#                       return_sequences=True))
#        model.add(Dropout(0.5))
#        model.add(LSTM(lstm_nodes, 
#                       return_sequences=False))
#        model.add(Dropout(0.5))
#        model.add(Dense(n_classes, activation='softmax'))
#        model.summary()
#        
#        #compile model 
#        model.compile(optimizer='adam', loss='categorical_crossentropy', 
#              metrics=['accuracy'])
#
#        #train model
#        history = model.fit(train_x_mbs, train_y_mbs, epochs=1, 
#                        batch_size=60, shuffle=True, verbose=1)
#        #note that we create a new model each epoch, so 'epochs' = 1