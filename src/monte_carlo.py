import os
import pandas as pd
from tqdm import tqdm
import numpy as np 
from src.coxphm import CoxProportionalHazardsModel
from lifelines import CoxPHFitter
from src.files import set_directories, get_indexes, save_indexes, save_model, copy_and_paste_indexes
from src.rubins_rule import * 
from scipy.interpolate import interp1d

class Monte_Carlo_Cross_Validation:
    """
    This class implements Monte Carlo Cross-Validation for time-to-event modeling, specifically for 
    Cox Proportional Hazards and Fine-Gray models. It facilitates repeated random subsampling 
    validation and allows for combined models through Rubin's rule for combining multiple imputations.

    Attributes:
        model: The survival model to be used (either Cox Proportional Hazards or Fine-Gray).
        combined_models: Boolean flag indicating whether to combine models using Rubin's rule.
        t_star: An object containing survival analysis settings, including time and event columns.
        sufix: Suffix for file paths based on the time-event data.
        predictors: List of predictors used in the model.
        repetitions: Number of repetitions for cross-validation.
        duration_col: Name of the column representing the event duration.
        event_col: Name of the column representing the event indicator (1 if event occurred, 0 otherwise).
    """
    def __init__(self, model, t_star, repetitions=10, combined_models=True, 
                 penalizer=0.0, internal_testing_proportion= 0.2):
        """
        Initializes the Monte Carlo Cross-Validation object with the specified model and parameters.

        Args:
            model: The survival model to use (e.g., Cox Proportional Hazards or Fine-Gray).
            t_star: An object containing survival analysis settings, including event time and predictors.
            repetitions (int): Number of cross-validation repetitions (default is 10).
            combined_models (bool): If True, combines models using Rubin's rule (default is False).
        """
        self.model = model
        self.combined_models = combined_models
        self.t_star = t_star
        self.sufix = t_star.dir_name
        self.predictors = t_star.predictors
        self.repetitions = repetitions
        self.duration_col = self.t_star.time_event[0]
        self.event_col = self.t_star.time_event[1]
        self.environment = model.environment
        self.train_means = None
        self.penalizer = penalizer
        self.prediction_date = int(365.25*5)
        self.internal_testing_proportion = internal_testing_proportion
             
    def check_validity(self, sufix):     
        source_path = [f for f in os.listdir('data/') 
                    if os.path.isdir(os.path.join('data/', f)) 
                    and self.t_star.indexes_from in f][0]
        
        folders = len([f for f in os.listdir('data/'+source_path) 
                   if os.path.isdir(os.path.join('data/'+source_path, f))])
        
        are_equal = source_path.split('_ID_')[0] == sufix.split('_ID_')[0]
        
        if not are_equal: 
            print(are_equal)
            print('Source configuration: ', source_path.split('_ID_')[0])
            print('New configuration:', sufix.split('_ID_')[0])
            raise ValueError("The tStar configuration must be equal to use indexes")

        if self.repetitions!=folders:
            print('Repetitions of source configuration: ', folders)
            print('Repetitions of new configuration: ', self.repetitions)
            
            raise ValueError('The ID for source indexes must have the same number of repetitions.')
        
    def fit_models_and_save_in_list(self, files, train_idx):
        models = []  
                
        for file in tqdm(files, desc="Fitting and saving models..."):
            df = pd.read_csv('data/'+self.sufix+"/"+file)
            df.set_index('index', inplace=True)

            self.train_means = df.loc[train_idx].mean()      
            
            new_model = CoxPHFitter(penalizer=self.penalizer)
            new_model.fit(df.loc[train_idx][[self.duration_col, self.event_col] + 
                        self.t_star.predictors], 
                        duration_col=self.duration_col, 
                        event_col=self.event_col, 
                        show_progress=False)        
            
            models.append(new_model)        
            
        return models

    def predict_with_combined_model(self, files, idx, full_dir_path_pred, new_cox_model): 
        
        """
        Makes predictions using the combined model for each test file and saves the results.

        Args:
            files (list): List of file names containing data.
            idx (list): Indexes of test samples.
            full_dir_path_pred (str): Directory path to save predictions.
        """
        for file in tqdm(files, desc="Fitting and saving models..."):
                
                number = file.split('_')[-1].replace('.csv', '')
                path_file_to_predict = 'data/'+self.sufix+"/"+file
                df = pd.read_csv(path_file_to_predict)
                df.set_index('index', inplace=True)

                data_test = df.loc[idx][self.t_star.predictors]   
                prediction_path = full_dir_path_pred+'/prediction_model_'+str(number)+'_combined_.csv' 
                
                _ = self.predict(data_test, prediction_path, new_cox_model)
                    
    def combine_monte_carlo_predictions(self, files, files_test, train_idx, validation_idx, 
                                        test_idx, full_dir_path_r, full_dir_path_pred): 
        """
        Combines multiple models using Rubin's rule and makes predictions with the combined model.

        Args:
            files (list): List of training files.
            files_test (list): List of test files.
            train_idx (list): Training data indexes.
            validation_idx (list): Validation data indexes.
            test_idx (list): Test data indexes.
            full_dir_path_r (str): Directory path to save combined models.
            full_dir_path_pred (str): Directory path to save predictions.
        """
        if isinstance(self.model, CoxProportionalHazardsModel): 
            # Fit the model for each training dataset and save the resulting models in a list.
            models = self.fit_models_and_save_in_list(files, train_idx)
            
            # Combine the models using Rubin's rule to create a new combined Cox Proportional Hazards model.
            # Rubin's rule is applied to the list of models, the suffix for file paths, the files, 
            # the time-event data, and the model's environment (e.g., Python or R).
            
            new_cox_model = apply_rubins_rule(models, 
                                              self.sufix,
                                              files,
                                              self.t_star.time_event,
                                              self.model.environment)
            
            # Save the combined model to the specified directory (full_dir_path_r) for future reference.
            save_model(self.model.environment, full_dir_path_r, new_cox_model)
            
            # Make predictions using the combined model on the validation dataset (or internal validation)
            self.predict_with_combined_model(files, 
                                             validation_idx, 
                                             full_dir_path_pred, 
                                             new_cox_model)
            
            # Make predictions using the combined model on the test dataset (after t_star or external validation)
            # and save the results.
            path_to_save_prediction_prediction = os.path.join(full_dir_path_pred, 'after_tstar')
            os.makedirs(path_to_save_prediction_prediction, exist_ok=True)
        
            self.predict_with_combined_model(files_test, 
                                             test_idx, 
                                             path_to_save_prediction_prediction, 
                                             new_cox_model)
        
     
    def predict(self, new_data,  file_name, new_model, time=None, save=True):
        """
        Predict the survival probability using the fitted Cox Proportional Hazards model.

        Parameters:
        new_data (pd.DataFrame): The new data to predict on.
        time (float, optional): The specific time point to predict survival probability at. If not provided, the function will return survival probabilities across all times.

        Returns:
        predictions: The predicted survival probabilities.
        """
        
        if file_name is None: 
            ValueError("You must provide a file name")
        
        if self.environment == "python":
            if time is not None:
                predictions_df = self.custom_predict_survival_function(new_model, new_data, [self.prediction_date])
                if save:
                    predictions_df.to_csv(file_name)
                else: 
                    return predictions_df
            else:
                if save: 

                    predictions_df = self.custom_predict_survival_function(new_model, new_data, [self.prediction_date])

                    predictions_df.columns = new_data.index
                    
                    predictions_df.to_csv(file_name)

                    del new_model
                    
                return predictions_df
             
    def custom_predict_survival_function(self, model, new_data, times):
        
        coefficients = model.params_
        baseline_survival = model.baseline_survival_
        
        # Align new_data columns with coefficients
        new_data = new_data[coefficients.index].astype(float) - self.train_means[coefficients.index]
        
        # Ensure coefficients are in the correct shape
        coefficients = coefficients.values  # Convert to NumPy array
        
        # Calculate linear predictors
        linear_predictors = np.dot(new_data.values, coefficients)  # Shape: (n_samples,)
            
        baseline_times = baseline_survival.index.values.astype(float)
        baseline_surv_values = baseline_survival.values.flatten()
        
        # Create interpolation function
        interp_func = interp1d(baseline_times, baseline_surv_values, kind='linear', fill_value="extrapolate", assume_sorted=True)
        
        # Initialize DataFrame for survival functions
        survival_df = pd.DataFrame(index=times)
        
        # Compute survival probabilities
        for i, lp in enumerate(linear_predictors):
            # Interpolated baseline survival at desired times
            interpolated_baseline_survival = interp_func(times)
            # Survival function for individual i
            survival_prob = interpolated_baseline_survival ** np.exp(lp)
            survival_df[i] = survival_prob
        
        return survival_df
                    
    def run(self):
        """
        Executes the Monte Carlo cross-validation process, including model fitting, validation, 
        and predictions for each repetition. Results are saved to appropriate directories.
        """
        # Extract the suffix for file directories by removing 'test_' and 'train_' from the file name.
        sufix = self.t_star.file_name_testing.replace('test_','').replace('train_','')
        
        # Get a list of training and test files from the data directory that correspond to the suffix.
        files_with_train = [f for f in os.listdir('data/'+sufix) if 'Train' in f]
        files_with_test = [f for f in os.listdir('data/'+sufix) if 'Test' in f]
        
        # Loop through each repetition for the Monte Carlo cross-validation.
        for r in range(self.repetitions):
            # Set up directories for saving predictions, data, and models for this repetition.
            full_dir_path_pred, full_dir_path_data, full_dir_path_r = set_directories(sufix, r)
                
            if self.t_star.indexes_from == None:
                
                # Get the path of the first training and test file to retrieve indexes for training, validation, and testing.
                path_to_get_indexes = 'data/'+sufix+"/"+files_with_train[0]
                path_to_get_indexes_test = 'data/'+sufix+"/"+files_with_test[0]
                
                # Load the training and validation indexes from the first training file.
                train_idx, validation_idx = get_indexes(path_to_get_indexes, 
                                                        test_proportion=self.internal_testing_proportion)
                
                # For the test set, load the indexes directly from the first test file.
                # We only need the test set indexes, not the data itself, at this point.
                test_idx = pd.read_csv(path_to_get_indexes_test).set_index('index').index
                unuseful_idx = None
                
                # Save the training and validation indexes to the appropriate data directory.
                save_indexes(train_idx, validation_idx, full_dir_path_data)
                
                # Save the test set indexes separately in the 'after_t_star' directory.
                save_indexes(unuseful_idx, test_idx, full_dir_path_data+'/after_t_star/')
            
            else:
                #check if indexes can be copied from source folder.
                self.check_validity(sufix)
                 
                #copy and paste indexes for each repetition from source folder to current folder               
                train_idx, validation_idx, test_idx, unuseful_idx = copy_and_paste_indexes(self.t_star, 
                                                                                            sufix, r) 
                
            # If the combined_models flag is set, apply Rubin's rule to combine the models from different imputations
            # in each repetition.
            if self.combined_models:         
                self.combine_monte_carlo_predictions(files_with_train, 
                                                     files_with_test, train_idx, 
                                                     validation_idx, 
                                                     test_idx, full_dir_path_r, 
                                                     full_dir_path_pred)
            
            # Otherwise, perform individual model predictions without combining them. 
            else: 
                self.individual_monte_carlo_prediction(files_with_train, 
                                                       files_with_test, train_idx, 
                                                       validation_idx, 
                                                       test_idx, full_dir_path_r, 
                                                       full_dir_path_pred)
                        
    def individual_monte_carlo_prediction(self, files_with_train, files_with_test, train_idx, validation_idx, 
                                        test_idx, full_dir_path_r, full_dir_path_pred): 
        """
        Performs individual model predictions for each Monte Carlo repetition without combining models.

        Args:
            files_with_train (list): List of training files.
            files_with_test (list): List of test files.
            train_idx (list): Training data indexes.
            validation_idx (list): Validation data indexes.
            test_idx (list): Test data indexes.
            full_dir_path_r (str): Directory path to save individual models.
            full_dir_path_pred (str): Directory path to save predictions.
        """
        
        for file in tqdm(files_with_train, desc="Fitting and saving models..."):
            df = pd.read_csv('data/'+self.sufix+"/"+file)
            
            self.model.fit(data=df.loc[train_idx], duration_col=self.duration_col, 
                           event_col=self.event_col, predictors=self.predictors)
            
            number = file.split('_')[-1].replace('.csv', '')
            
            self.model.save_model(file_path= full_dir_path_r+'/'+ self.model.environment+'_'+number)
            
            self.validation_prediction(files_with_train, full_dir_path_pred,  model_number= number, dataset='validation',
                                       test_idx = validation_idx)
            #We are not predicting on training set, to do that we have to pass train_idx instead of validation idx 
            # and we should modify the dataset argument 'training' 
            
            new_path = full_dir_path_pred+'/after_t_star/'
            self.validation_prediction(files_with_test, new_path, model_number= number, dataset='test', 
                                       test_idx = test_idx)

    def validation_prediction(self, files, full_dir_path_pred, model_number=None, dataset=None, test_idx=None): 
        """
        Makes predictions for the test data and saves the results to the specified directory.

        Args:
            files (list): List of file names for which predictions need to be made.
            full_dir_path_pred (str): Directory path to save predictions.
            test_idx (list): Indexes of the test samples.
        """
        os.makedirs(full_dir_path_pred, exist_ok=True)
        
        for file in tqdm(files, desc="Fitting and saving models..."):
            path = 'data/'+self.sufix+"/"+file
            
            df = pd.read_csv(path)
            df.set_index('index', inplace=True)
            id_file = file.split('_')[-1].replace('.csv', '')
                            
            df = df.loc[test_idx]
            
            prediction_path = (full_dir_path_pred+'/'+dataset+'_prediction_model_'+str(model_number)+
                                '_imputed_'+str(id_file)+'.csv')
            
            #_ = self.model.predict(new_data=df, file_name=prediction_path)

            
            if isinstance(self.model, CoxProportionalHazardsModel):
                _ = self.model.predict(new_data=df, file_name=prediction_path)
                
            else: 
                raise TypeError('Model type {} is not supported'.format(type(self.model)))
        
    
                
                