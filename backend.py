import pandas as pd
import numpy as np
from validation import Validation as Val
import os


def list_csv_files():
    #Creates and print a list of CSV files existing in folder
    csv_files = [file for file in os.listdir() if file.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the current directory.")
    else:
        print("CSV files in the current directory:")
        for csv_file in csv_files:
            print(csv_file)

    return csv_files


def read_in_csv(csv_filename:str):
    # Read in CSV filename, creates a Data Frame and print out header
    df = pd.read_csv(f"{csv_filename}")
    print("*************** HEAD ***************")
    print(df.head())
    print("************************************")
    print(f"Columns: {df.columns}")
    return df

def check_df_missing_values(df):
    # Check the Data Frame for missing values
    # If there is missing values, 
    # print numbers missing for each column and exit
    missing_values = df.isnull().sum()

    columns_with_missing_values = missing_values[missing_values > 0]

    if columns_with_missing_values.empty:
        print("No missing values found in the DataFrame.")
    else:
        print("Missing values in the DataFrame:")
        print(columns_with_missing_values)
        print("Please choose another csv file and start over!")
        exit()

def check_df_dtypes(df):
    # Check DF for columns of class 'object',
    # returns dummies_needed True if so
    dummies_needed = False
    object_columns = df.select_dtypes(include=['object']).columns

    if not object_columns.empty:
        print("DataFrame has columns of dtype 'object'.")
        dummies_needed = True
        return dummies_needed
    else:
        print("DataFrame doesn't have columns of dtype 'object'.")


def create_dummies(df):
    # Create dummies and print out new column names, returns DF with dummies
    df = pd.get_dummies(df,drop_first=True,dtype='int8')
    print("Dummies Created!")
    print(f"New column names: {df.columns}")
    return df


def target_column(df,target_column_name):
        # split DF in Label and Features, returns X,y
        X = df.drop(f"{target_column_name}",axis=1)
        y = df[f'{target_column_name}']
        return X,y


def split_data(X,y):
    # Splitting X,y depending on size of X,
    # returns X_train,X_test,y_train,y_test
    if len(X) >= 2000:
        split_size = 0.1
    elif 500 < len(X) < 2000:
        split_size = 0.2
    else:
        split_size = 0.33
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=split_size,
                                                        random_state=101)
    print(f'Train Test Split Complete with split size {split_size}')
    return X_train, X_test, y_train, y_test

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,LinearRegression,LogisticRegression,Ridge,ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,SVR

# Creating regressors with GridSearchCV and returns the best model

def create_lir_model_cv(X_train,y_train):
    scaler = StandardScaler()
    lir = LinearRegression()
    param_grid = {}
    operations = [('scaler',scaler),('lir',lir)]
    pipe_lir = Pipeline(operations)
    lir_best_model_cv = GridSearchCV(pipe_lir,
                      cv=10,
                      scoring="neg_mean_squared_error",
                      param_grid=param_grid)
    lir_best_model_cv.fit(X_train,y_train)
    return lir_best_model_cv

def create_lasso_model_cv(X_train,y_train):
    scaler = StandardScaler()
    lasso = Lasso()
    param_grid = {"lasso__alpha" : [.001,.01,1,5,10]}
    operations = [('scaler',scaler),('lasso',lasso)]
    pipe_lasso = Pipeline(operations)
    lasso_best_model_cv = GridSearchCV(pipe_lasso,
                      cv=10,
                      scoring="neg_mean_squared_error",
                      param_grid=param_grid)
    lasso_best_model_cv.fit(X_train,y_train)
    return lasso_best_model_cv

def create_ridge_model_cv(X_train,y_train):
    scaler = StandardScaler()
    ridge = Ridge()
    param_grid = {"ridge__alpha" : [.001,.01,1,5,10]}
    operations = [('scaler',scaler),('ridge',ridge)]
    pipe_ridge = Pipeline(operations)
    ridge_best_model_cv = GridSearchCV(pipe_ridge,
                      cv=10,
                      scoring="neg_mean_squared_error",
                      param_grid=param_grid)
    ridge_best_model_cv.fit(X_train,y_train)
    return ridge_best_model_cv

def create_elastic_model_cv(X_train,y_train):
    scaler = StandardScaler()
    elastic = ElasticNet()
    param_grid = {'elastic__alpha' : [0.1,1,5,10,50,100],
                  "elastic__l1_ratio" : [.1, .5, .7,.9, .95, .99, 1]}
    operations = [('scaler',scaler),('elastic',elastic)]
    pipe_elastic = Pipeline(operations)
    elastic_best_model_cv = GridSearchCV(pipe_elastic,
                      cv=10,
                      scoring="neg_mean_squared_error",
                      param_grid=param_grid)
    elastic_best_model_cv.fit(X_train,y_train)
    return elastic_best_model_cv

def create_svr_model_cv(X_train,y_train):
    scaler = StandardScaler()
    svr = SVR()
    param_grid = {"svr__kernel" : ['linear', 'poly', 'rbf', 'sigmoid'],
              "svr__degree" : np.arange(1,4),
              "svr__C" : np.logspace(0,1,10),
              "svr__gamma" : ['scale','auto'],
              "svr__epsilon"  : [0,0.001,0.1,0.5,1,2,2.1,3]}
    operations = [('scaler',scaler),('svr',svr)]
    pipe_svr = Pipeline(operations)
    svr_best_model_cv = GridSearchCV(pipe_svr,
                      cv=10,
                      scoring="neg_mean_squared_error",
                      param_grid=param_grid)
    svr_best_model_cv.fit(X_train,y_train)
    return svr_best_model_cv

# Print out scores of all models and pick the best one
# depending on highest r2score

def get_regressor_scores(all_models:list,X_test,y_test):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    best_model = None
    best_r2_score= 0

    for model in all_models:
        y_pred = model.predict(X_test)
        MAE = mean_absolute_error(y_test,y_pred)
        MSE = mean_squared_error(y_test,y_pred)
        RMSE = np.sqrt(MSE)
        model_r2_score = r2_score(y_test, y_pred)
        print(model.best_estimator_)
        print(f'Best params: {model.best_params_}')
        print(f'MAE: {MAE}')
        print(f'RMSE: {RMSE}')
        print(f'r2score: {model_r2_score}')
        print(f'********************\n')
        if model_r2_score > best_r2_score:
            best_r2_score = model_r2_score 
            best_model = model
    return best_model,best_r2_score

def train_and_dump(best_model,X,y,option):
    # Train final model on entire DF and dumping 
    # it as Best Regressor or Best Classifier depending on option
    from joblib import dump
    print("Training of full dataset, please be patient....")
    best_model.fit(X,y)
    if option == "Regressor":
        dump(best_model,"Best_Regressor.joblib")
        print("The model is now saved as Best_Regressor.joblib")
    if option == "Classifier":
        dump(best_model,"Best_Classifier.joblib")
        print("The model is now saved as Best_Classifier.joblib")


# Creating Classifiers with GridSearchCV and returns the best model

def create_lor_model_cv(X_train,y_train):
    scaler = StandardScaler()
    lor = LogisticRegression()
    param_grid = {'lor__C': np.logspace(0,4,10)}
    operations = [('scaler',scaler),('lor',lor)]
    pipe_lor = Pipeline(operations)
    lor_best_model_cv = GridSearchCV(pipe_lor,
                      cv=10,
                      scoring="accuracy",
                      param_grid=param_grid)
    lor_best_model_cv.fit(X_train,y_train)
    return lor_best_model_cv

def create_knn_model_cv(X_train,y_train):
    scaler = StandardScaler()
    knn = KNeighborsClassifier()
    param_grid = {"knn__n_neighbors":list(range(1,30))}
    operations = [('scaler',scaler),('knn',knn)]
    pipe_knn = Pipeline(operations)
    knn_best_model_cv = GridSearchCV(pipe_knn,
                      cv=10,
                      scoring="accuracy",
                      param_grid=param_grid)
    knn_best_model_cv.fit(X_train,y_train)
    return knn_best_model_cv

def create_svc_model_cv(X_train,y_train):
    scaler = StandardScaler()
    svc = SVC()
    param_grid = {'svc__C':np.logspace(0,1,10),
              'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              "svc__degree": np.arange(1,6),
              "svc__gamma":['scale', 'auto'],
              }
    operations = [('scaler',scaler),('svc',svc)]
    pipe_svc = Pipeline(operations)
    svc_best_model_cv = GridSearchCV(pipe_svc,
                      cv=10,
                      scoring="accuracy",
                      param_grid=param_grid)
    svc_best_model_cv.fit(X_train,y_train)
    return svc_best_model_cv

def get_classifier_scores(all_models:list,X_test,y_test):
    # Takes in a list of models, printing scores, and returning the
    # best model based on highest accuracy
    from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
    best_model = None
    best_accuracy= 0

    for model in all_models:
        y_pred = model.predict(X_test)
        
        model_accuracy = accuracy_score(y_test,y_pred)
        print(model.best_estimator_)
        print(f'Best params: {model.best_params_}')
        print(f'Confusion Matrix: {confusion_matrix(y_test,y_pred)}')
        print(f'Classification Report: {classification_report(y_test,y_pred)}')
        print(f'********************\n')
        if model_accuracy > best_accuracy:
            best_accuracy = model_accuracy 
            best_model = model
    return best_model,best_accuracy