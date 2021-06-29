Purpose: The purpose of this project is first to define, train and test two 
         XGBoost models with features to predict 10-year CV mortality, and 
         second to interpret these two models using the SHAP library to 
         generate insights into which features and their respective values 
         lead to patients developing higher risks of CV mortality.

Date: June 28, 2021

This project consists of two files:
- CACConsortium_Functions.py: Defines relevant functions.
- CACConsortium_Xgboost_Shap_Main.ipynb: Runs the functions in the previous 
                                         file to accomplish the objective 
                                         defined in "Purpose" above.

Functions in CACConsortium_Functions.py:
- get_np(df, feature_names, out_var): Converts input DataFrame to a NumPy array.
- load_processed_features(str_file_postfix): Loads training and test data.
- setup_classifier_training(train_data, train_label, seed_random): 
  Initializes and tunes the model using cross-validation.
- fit_mod(train_data, train_label, model, model_post_filename, algo_name='',
          toggle_perform_training=1, toggle_save_trained_model=1): 
  Fits the model to the training data.
- get_roc_auc_values(model, test_data, test_label, classifier_threshold): 
  Calculates the classifier metrics for the model.

The code in CACConsortium_Xgboost_Shap_Main.ipynb does the following:
- Runs each of the functions from CACConsortium_Functions.py.
- Generates SHAP values for all patients in the training data set for 
  the given model. 
- Creates SHAP summary and dependence plots. 

The models are saved and loaded using Pickle in Python. Please refer to the 
SHAP github repository for documentation and instruction on how to use SHAP.
