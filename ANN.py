from validation import Validation as Val
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from joblib import dump,load
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,mean_absolute_error,\
mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.utils import to_categorical

# Create lists of possible values for our variables
# Easy to make both validations and to get the desired value if needed

activation_list = ["relu","sigmoid","softmax","tanh"]
loss_list = ["mse","binary_crossentropy","categorical_crossentropy"]
optimizer_list =  ["adam","rmsprop","sgd"]
monitor_list = ["val_loss","accuracy"]
mode_list = ["auto","min","max"]
verbose_list = [0,1,2]

class MyAnn():
    '''
    A class to create parameters for ANN (Artificial Neural Network),
    Note! method: build_ann must be called to create the network

    Parameters:
    ----------
    data_set (str): Valid absolute path to .csv dataset file,  
    Data cleaned and ready to process in csv dataset form. 

    target (str): Valid name to the Labels' column.

    hidden_layer_size (tuple): A Tuple like to describe the shape of hidden layers, default= (100,) 
    The number in each tuple field represents the number of neurons in the related hidden layer. 
    In case it was negative, so it is dropout layer and the value represent the percent. 

    activation (str): Valid Activation function for the hidden layer default is "relu",
    other options  "sigmoid", "softmax", "tanh".

    loss (str): Valid Loss function, values = "mse", "binary_crossentropy" or "categorical_crossentropy"

    optimizer (str): Valid Optimizer for the output layer, the default is "adam"
    other options for this class is "rmsprop" and "sgd".

    batch_size (int): Integer or None. Number of samples per gradient update. 
    If unspecified, batch_size will default to 32.

    epochs (int): Number of epochs to train the model, default is 1 epoch.

    monitor (str): Result to be monitored, default = "val_loss", other option "accuracy"

    patience (int): Number of epochs with no improvement after which training will be stopped,
    default = as value in epochs.

    mode (str): One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored 
    has stopped decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing;
    in "auto" mode, the direction is automatically inferred from the name of the monitored quantity. 

    verbose (int): 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    Default = 1.

    use_multiprocessing (bool): If True, use process-based threading. If unspecified, 
    use_multiprocessing will default as False.
    '''
    def __init__(self,
                 data_set:str,
                 target:str,
                 loss:str,
                 monitor:str,
                 mode:str,
                 patience:int=None,
                 hidden_layer_sizes:tuple = (100,),
                 activation:str="relu",
                 optimizer:str="adam",
                 batch_size:int=32,
                 epochs:int=1,
                 verbose:int=0,
                 use_multiprocessing:bool=False
                 ) -> None:

        if Val.validate_csv_filename(data_set):
            self.data_set = data_set
            self.df = pd.read_csv(data_set)
            Val.validate_df(self.df)
        else:
            raise ValueError("[i] Bad value: data_set")
        if Val.validate_str(target) and target in self.df.columns:
            self.target = target
        else: 
            raise ValueError("[i] Bad value: target")
        if Val.validate_tuple:
            self.hidden_layer_sizes = hidden_layer_sizes
        else:
            raise ValueError("[i] Bad value: hidden_layer_sizes")
        if Val.validate_str(activation) and activation in activation_list:
            self.activation = activation
        else:
            raise ValueError("[i] Bad value: activation")
        if Val.validate_str(loss) and loss in loss_list:
            self.loss = loss
        else: 
            raise ValueError("[i] Bad value: loss")
        if Val.validate_str(optimizer) and optimizer in optimizer_list:
            self.optimizer = optimizer
        else: 
            raise ValueError("[i] Bad value: optimizer")   
        if Val.validate_int(batch_size):
            self.batch_size = batch_size
        else: 
            raise ValueError("[i] Bad value: batch_size")    
        if Val.validate_int(epochs):
            self.epochs = epochs
        else: 
            raise ValueError("[i] Bad value: epochs")
        if Val.validate_str(monitor) and monitor in monitor_list:
            self.monitor = monitor
        else: 
            raise ValueError("[i] Bad value: monitor")
        if Val.validate_int(patience):
            self.patience = patience
        elif self.patience == None:
            self.patience = epochs
        else: 
            raise ValueError("[i] Bad value: patience")
        if Val.validate_str(mode) and mode in mode_list:
            self.mode = mode
        else:
            raise ValueError("[i] Bad value: mode")
        if Val.validate_int(verbose) and verbose in verbose_list:
            self.verbose = verbose
        else:
            raise ValueError("[i] Bad value: verbose")    
        if Val.validate_bool(use_multiprocessing):
            self.use_multiprocessing = use_multiprocessing
        else:
            raise ValueError("[i] Bad value: use_multiprocessing")
        
        # Create Features and Label, save feature columns before turning X to array 
        self.features_= self.df.drop(self.target,axis=1).columns
        self.X = self.df.drop(self.target,axis=1).values
        self.classes_ = self.target
        self.y = self.df[self.target].values

        # Create a string that represent the operation we want for our target
        self.type_of_operation = type_of_target(self.y)

        # Train Test Split
        if len(self.X) < 800:
            self.split_size = 0.3
        elif len(self.X) > 8000:
            self.split_size = 0.1
        else:
            self.split_size = 0.2

        # Train Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split\
            (self.X, self.y, test_size=self.split_size,random_state=101)

        # Scaling
        self.scaler = MinMaxScaler()
        self.scaled_X_train = self.scaler.fit_transform(self.X_train)
        self.scaled_X_test = self.scaler.transform(self.X_test)

        # Create Early stop
        self.early_stop = EarlyStopping(monitor = self.monitor,
                           patience = self.patience,
                           mode = self.mode,
                           restore_best_weights=True)
        
        # Create a variable for number of categories
        self.num_cat = np.unique(self.y).size
        
        # If multiclass, we convert y to categorical based on # categories
        if self.type_of_operation == "multiclass":
            self.y_train = to_categorical(self.y_train,
                                        num_classes = self.num_cat)
            self.y_test = to_categorical(self.y_test,
                                        num_classes = self.num_cat)
        
        # Declare variables needed 
        self.out_activation_ = None
        self.n_nodes_ = None
        self.n_layers_ = None   
        self.n_outputs_ = 1 # We set default as 1 and change only if multi-class
        self.loss_ = None
        self.best_loss_ = None    
        self.model = None
        self.history = None
        self.y_pred = None
        self.model_loss_df = None
    
    def build_ann(self):
        self.model = Sequential()
        # Create Input Layer with #Neurons equal to #Features
        self.model.add(Dense(units = self.scaled_X_train.shape[1],
                        activation = self.activation))
        
        # Create Hidden Layers, for negative values create dropout
        for value in self.hidden_layer_sizes:
            if value > 0:
                self.model.add(Dense(
                    units = value,
                    activation = self.activation))
            elif -1 < value < 0:
                self.model.add(Dropout(abs(value))) # make value positive

        # Create Output Layer depending on type of target column
        if self.type_of_operation == "multiclass":

            # Output-neurons equal to #target categories, softmax activation
            self.model.add(Dense(units = self.num_cat,
                                 activation = "softmax"))
            self.out_activation_ = "softmax"
            # Change n_outputs_ to number of categories instead of default (1)
            self.n_outputs_ = self.num_cat

        elif self.type_of_operation == "binary":
                        # 1 Neuron with sigmoid activation for binary
            self.model.add(Dense(units = 1,
                                 activation = "sigmoid"))
            self.out_activation_ = "sigmoid"

        elif self.type_of_operation == "continuous":
            # 1 Neuron with Linear activation (Default) for regressor
            self.model.add(Dense(units = 1))
            self.out_activation_ = "linear"

        else:
            raise ValueError("[i] Target column not compatible with this class")
            
        # Compile
        if self.type_of_operation in ["binary","multiclass"]:
            metrics = "accuracy"
        else:
            metrics = "mse"
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=[metrics])
        # Train   
        self.history = self.model.fit(x = self.scaled_X_train,
                                      y = self.y_train,
                                      epochs = self.epochs,
                                      batch_size = self.batch_size,
                                      verbose = self.verbose,
                                      validation_data = (self.scaled_X_test,
                                                         self.y_test),
                                      callbacks = [self.early_stop])
                                    # use_multiprocessing, can't make it 
                                    # work with this version of keras
        # Make predictions
        if self.type_of_operation == "binary":
            # y_pred for binary will be 1 or 0.(True/False for values over 0.5)
            self.y_pred = (self.model.predict(self.scaled_X_test)>0.5)
        elif self.type_of_operation == "multiclass":
            # this will give us an array with int values for corresponding prediction
            self.y_pred = np.argmax(self.model.predict(self.scaled_X_test),axis=-1)
            # Here we change our int values to categorical representation
            # to match the array of our true data
            self.y_pred = to_categorical(self.y_pred, num_classes=self.num_cat)

        else:
            # y_pred -> continuous values
            self.y_pred = self.model.predict(self.scaled_X_test)

        # To get a number of all nodes in this network, we sum upp all positive
        # values from hidden_layer_sizes and add the input and output layer
        self.n_nodes_ = sum(x for x in self.hidden_layer_sizes if x > 0)+\
            self.scaled_X_train.shape[1]+self.n_outputs_
        # Not sure if I should take account for the dropout,
        # what is the correct approach ?? 
        
        # To get #Layers, we sum up the hidden layers and add 2 (in/out)
        self.n_layers_ = len(self.hidden_layer_sizes)+2
        loss_history = self.history.history['loss']
        # Here dropout will count as a Layer

        # declare best_loss_ as the min loss value during fit
        self.best_loss_ = min(loss_history)
        # declare the loss_ as the last value loss during fit
        self.loss_ = loss_history[-1]

    def model_loss(self):
        '''Returns a Pandas Data frame with loss and val_loss for Regressor.
            If Categorical, also includes accuracy and val_accuracy'''
        self.model_loss_df = pd.DataFrame(self.history.history)
        return self.model_loss_df
    
    def plot_model_loss(self):
        '''Plot loss and val_loss'''
        self.model_loss_df[["loss","val_loss"]].plot()

    def plot_model_accuracy(self):
        '''Plot accuracy and val_accuracy'''
        if self.type_of_operation == "binary" or "multiclass":
            self.model_loss_df[["accuracy","val_accuracy"]].plot()
        else:
            print(f"Method not available for {self.type_of_operation} label.")

    def model_predict(self):
        '''Predicts a random row from test data, 
            printing out prediction and true value for that row'''
        test_idx = np.random.randint(0,len(self.scaled_X_test))
        # Add an extra dimension to get the correct format for predict
        test_pred = test_pred = self.model.predict(np.expand_dims\
                                                   (self.scaled_X_test[test_idx],
                                                    axis=0))
        if self.type_of_operation == "binary":
            test_pred = (test_pred > 0.5).astype(int)
        elif self.type_of_operation == "multiclass":
            test_pred = np.argmax(test_pred,axis=-1)
            test_pred = to_categorical(test_pred,num_classes=self.num_cat)

        print(f"Predicted value from row {test_idx}: {test_pred}")
        print(f"True value from row {test_idx}: {self.y_test[test_idx]}")

    def save_model(self):
        '''Saves model and scaler'''
        # We convert current datetime object to string with Year Month Day 
        timestamp = datetime.datetime.now().strftime("%Y%m%d")

        dump(self.scaler,f"My_{self.type_of_operation}_Scaler_{timestamp}.pkl")

        # Here we could also train model on the full data before saving
        self.model.save(f"My_{self.type_of_operation}_Model_{timestamp}.h5")

        print(f"Scaler saved as: My_{self.type_of_operation}_Scaler_{timestamp}.pkl")
        print(f"Model saved as: My_{self.type_of_operation}_Model_{timestamp}.h5")

    @staticmethod
    def load_model(filename_model,filename_scaler):
        '''Load model and scaler. Inputs: filename_model, filename_scaler.
        Returns: loaded_model, loaded scaler'''
        try:
            loaded_scaler = load(filename_scaler)
            loaded_model = load_model(filename_model)
            print(f"Loaded model and scaler")
            return loaded_model,loaded_scaler
        except Exception as e:
            print(f"Failed to load. Error: {e}")

    def print_classification_report(self):
        '''Prints classification_report'''
        if self.type_of_operation == "binary" or "multiclass":
            print(classification_report(y_true=self.y_test,y_pred=self.y_pred))
        else:
            print(f"Method not available for {self.type_of_operation} label.")

    def plot_predictions_scatter(self):
        '''Scatter-plot of true and predicted values '''
        # We set X as the indexes for our y-value/y_pred vectors
        plt.scatter(range(len(self.y_test)), self.y_test,
                         color='blue', label='True')
        plt.scatter(range(len(self.y_pred)), self.y_pred,
                         color='red', label='Predicted')
        plt.legend()  
        plt.show()

    def plot_residual_error(self):
        '''Dist-plot of residual error'''
        residual_error = abs(self.y_test-self.y_pred)
        plt.figure(figsize=(6,4),dpi=100)
        sns.distplot(residual_error)
        plt.ylabel("Residual Error")


    def rmse(self):
        '''Prints root mean squared error'''
        if self.type_of_operation == "continuous":
            print(mean_squared_error(y_true=self.y_test,y_pred=self.y_pred)**0.5)     
        else:
            print(f"Method not available for {self.type_of_operation} label.")

    def mae(self):
        '''Prints mean absolute error'''
        if self.type_of_operation == "continuous":
            print(mean_absolute_error(y_true=self.y_test ,y_pred= self.y_pred))     
        else:
            print(f"Method not available for {self.type_of_operation} label.")

    def r2score(self):
        '''Prints r2_score'''
        if self.type_of_operation == "continuous":
            print(r2_score(y_true=self.y_test ,y_pred= self.y_pred))     
        else:
            print(f"Method not available for {self.type_of_operation} label.")
    
