import os
import yaml
import pickle
import pandas as pd
from lifelines import CoxPHFitter
import numpy as np 
import time 

class CoxProportionalHazardsModel:
    """
    A class to implement the Cox Proportional Hazards Model in both Python and R environments.

    This class allows fitting, validating, and predicting using the Cox Proportional Hazards model. 
    It can work in either the Python environment (using the `lifelines` package) or in the R environment 
    (using the `survival` package via `rpy2`).

    Parameters:
    -----------
    environment : str, optional
        The environment in which the model is executed, either 'python' or 'R'. Default is 'python'.

    Attributes:
    -----------
    model : object
        The fitted model object, either from `lifelines.CoxPHFitter` (Python) or R's `coxph` model.
    
    path_models : str
        Path where models are saved, loaded from the `paths.yaml` file.

    path_imputed : str
        Path to imputed datasets, loaded from the `paths.yaml` file.
    
    penalizer : float
        It controls the overall penalty applied to the coefficients.
        Attach a penalty to the size of the coefficients during regression. This improves stability of the 
        estimates and controls for high correlation between covariates. For example, this shrinks the magnitude 
        value. See l1_ratio below. Alternatively, penalizer is an array equal in size to the number of parameters, 
        with penalty coefficients for specific variables. For example, penalizer=0.01 * np.ones(p) is the same as 
        penalizer=0.01        
    
    l1_ratio : float 
        It determines the proportion of L1 vs. L2 regularization.
        Specify what ratio to assign to a L1 vs L2 penalty. Same as scikit-learn. See penalizer above.

    Methods:
    --------
    fit(data=None, path=None, duration_col='T', event_col='E', predictors=None):
        Fit the Cox Proportional Hazards model to the provided data. Supports both Python and R environments.
    
    validate_proportionality(data=None, path=None):
        Validate the proportionality assumption of the Cox model using either Python or R.
    
    predict(new_data, time=None, save=True, file_name=None):
        Predict the survival probability for new data. Optionally save predictions to a CSV file.
    
    save_model(file_path):
        Save the fitted model to a file. Supports both Python and R environments.
    
    load_model(file_path):
        Load a previously saved model from a file.
    """
    def __init__(self, environment="python", penalizer=0.000, l1_ratio=0.000):
        """
        Initialize the CoxProportionalHazardsModel class.

        Parameters:
        environment (str): The environment to use for fitting and prediction ('python' or 'R'). Default is 'python'.
        """
        self.environment = environment
        self.model = None
        with open('paths.yaml', 'r') as file:
            paths = yaml.safe_load(file)
        
        self.path_models = paths['PATH_MODELS']
        self.path_imputed = paths['PATH_IMPUTED']  
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
    
    def fit(self, data=None, path=None, duration_col= 'T', 
                            event_col='E', predictors= None):
        """
        Fit the Cox Proportional Hazards model.

        Parameters:
        data (pd.DataFrame): The dataset containing the variables.
        duration_col (str): The name of the column representing the time duration.
        event_col (str): The name of the column representing the event occurrence (1 if event occurred, 0 otherwise).
        predictors (list): A list of predictor variables.

        Returns:
        None
        """
        if data is None and path is None: 
            raise ValueError("You must provide a data file or path to load it")
        
        if data is not None and path is not None:
            raise ValueError("You must provide a unique source for loading data")
        
        if predictors is None: 
            raise ValueError('You must add predictors')
        
        if path:
            data = pd.read_csv(self.path_imputed+path)
        
        if self.environment == "python":
            self.model = CoxPHFitter(penalizer=self.penalizer, 
                                    l1_ratio=self.l1_ratio)
            self.model.fit(data[[duration_col, event_col] + predictors], 
                            duration_col=duration_col, 
                            event_col=event_col, 
                            show_progress=False)
        else:
            raise ValueError("environment must be 'python'")
    
    def validate_proportionality(self, data = None, path = None):
        """
        Validate the proportionality assumption of the Cox model.

        Returns:
        test_results: The result of the proportional hazards test.
        """
        
        if data is None and path is None: 
            raise ValueError("You must provide a data file or path to load it")
        
        if data is not None and path is not None:
            raise ValueError("You must provide a unique source for loading data")
        
        if path: 
            data = pd.read_csv(path)
            
        if self.environment == "python":
            return self.model.check_assumptions(data, p_value_threshold=0.05)
        else:
            raise ValueError("environment must be 'python'")
    
    def predict(self, new_data, time = None, 
                save=True, file_name = None):
        """
        Predict the survival probability using the fitted Cox Proportional Hazards model.

        Parameters:
        new_data (pd.DataFrame): The new data to predict on.
        time (float, optional): The specific time point to predict survival probability at. 
        If not provided, the function will return survival probabilities across all times.

        Returns:
        predictions: The predicted survival probabilities.
        """
        if self.environment == "python":
            if time is not None:
                predictions_df = self.model.predict_survival_function(new_data, times=[time])
                if save:
                    if file_name is None: 
                        ValueError("You must provide a file name")
                    predictions_df.to_csv(file_name)
                else: 
                    return predictions_df
            else:
                if save: 
                    if file_name is None: 
                        ValueError("You must provide a file name")
                    predictions_df = self.model.predict_survival_function(new_data, 
                                             times= [i for i in range(1,3653, 14)])
                    predictions_df.to_csv(file_name)
                return predictions_df
        else:
            raise ValueError("environment must be 'python'")
 
    def save_model(self, file_path):
        """
        Save the fitted model to a file.

        Parameters:
        file_path (str): The file path where the model should be saved.

        Returns:
        None
        """
        model_name = str(file_path).replace('.csv', '.pkl')
        
        time.sleep(2)
        if self.environment == "python":
            with open(model_name, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            raise ValueError("environment must be 'python' or 'R'")
            
    def load_model(self, file_path):
        """
        Load a Cox Proportional Hazards model from a file.

        Parameters:
        file_path (str): The file path from which the model should be loaded.

        Returns:
        None
        """
        if self.environment == "python":
            with open(file_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise ValueError("environment must be 'python'")