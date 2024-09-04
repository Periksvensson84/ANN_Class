# Erik Svensson
import backend
from validation import Validation as Val
from sklearn.utils.multiclass import type_of_target

# Note: CSV-Files need to be in the same folder for this program to work!

option = Val.read_in_int_value_0_1("What operation wold you like to perform? "\
                                "press 1 for Regressor or 0 for Classifier: ")
if option == 1:
    print("You choose Regressor")
    option = "Regressor"
elif option == 0:
    print("You choose Classifier")
    option = "Classifier"

# Prints a list of CSV-files in the current directory
backend.list_csv_files()

csv_filename = Val.read_in_value(validation_function=Val.validate_csv_filename,
                                 message="Enter your CSV filename name without (\"\"): ")

df = backend.read_in_csv(csv_filename)

backend.check_df_missing_values(df)

dummies_needed = backend.check_df_dtypes(df)
if dummies_needed:
    input =  Val.read_in_int_value_0_1(
         "Press 1 to create dummies or 0 to exit: ")
    if input == 1:
        backend.create_dummies(df)
    else: exit()


target_column_name = Val.read_in_value(Val.validate_str_alnum,
                                       "Enter target column name without (\"\"): ")
if target_column_name in df.columns:
    X,y = backend.target_column(df,target_column_name)
    print("Features and Target column created (X,y)")
else:
        raise ValueError("[i] Target label name not in Dataframe")

X_train, X_test, y_train, y_test = backend.split_data(X,y)

# Create an empty list where we will append all our GridSearchCV models
all_models = []

# Check so that target column match the chosen operator
if option == "Regressor" and type_of_target(y) == "continuous":
    print("Creating models, please be patient.....")
    lir_best_model_cv = backend.create_lir_model_cv(X_train,y_train)
    all_models.append(lir_best_model_cv)
    lasso_best_model_cv = backend.create_lasso_model_cv(X_train,y_train)
    all_models.append(lasso_best_model_cv)
    ridge_best_model_cv = backend.create_ridge_model_cv(X_train,y_train)
    all_models.append(ridge_best_model_cv)
    elastic_best_model_cv = backend.create_elastic_model_cv(X_train,y_train)
    all_models.append(elastic_best_model_cv)
    svr_best_model_cv = backend.create_svr_model_cv(X_train,y_train)
    all_models.append(svr_best_model_cv)

    best_model, best_r2_score = backend.get_regressor_scores(all_models,X_test,y_test)
    print(f'The best model to use would be: \n {best_model.best_estimator_}\n')
    print(f'With an R2SCORE of: {best_r2_score}')

    input =  Val.read_in_int_value_0_1(
         "Do you agree that this model is the best and would like to Dump?\n"
              "Press 1 to dump model or 0 to exit: ")
    if input == 1:
        backend.train_and_dump(best_model,X,y,option)
    else: exit()


# Check so that target column match the chosen operator
elif option == "Classifier" and type_of_target(y) != "continuous": 
    print("Creating models, please be patient.....")
    lor_model = backend.create_lor_model_cv(X_train,y_train)
    all_models.append(lor_model)
    knn_model = backend.create_knn_model_cv(X_train,y_train)
    all_models.append(knn_model)
    svc_model = backend.create_svc_model_cv(X_train,y_train)
    all_models.append(svc_model)

    best_model, best_accuracy = backend.get_classifier_scores(all_models,X_test,y_test)
    print(f'The best model to use would be: \n {best_model.best_estimator_}\n')
    print(f'With an Accuracy of: {best_accuracy}')

    input =  Val.read_in_int_value_0_1(
         "Do you agree that this model is the best and would like to Dump?\n"
              "Press 1 to dump model or 0 to exit: ")
    if input == 1:
        backend.train_and_dump(best_model,X,y,option)
    else: exit()

else:
    print("Your target column doest match your option of Regressor/Classifier")
    exit()