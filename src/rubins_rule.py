import numpy as np 
import pandas as pd 
from lifelines import CoxPHFitter
from src.coxphm import CoxProportionalHazardsModel

def apply_rubins_rule(models, sufix, files_with_train, time_event_columns, environment): 
    
    pooled_beta, pooled_var_coef, m = pooled_betas(models, environment)

    pooled_baseline_survival, pooled_var_baseline, common_times = pool_baseline_function(models, m, environment)

    new_cox_model = empty_cox_model_initialization(sufix, files_with_train, time_event_columns, environment)

    new_cox_model = fill_empty_cox_model(new_cox_model, pooled_beta, pooled_var_coef, 
                                        pooled_baseline_survival, 
                                    common_times, pooled_var_baseline, environment)
    return new_cox_model

def pooled_betas(models, environment): 
    if environment=='python':
        try:
            betas = [model.model.params_ for model in models]
            vars_ = [model.model.variance_matrix_ for model in models]
        except: 
            betas = [model.params_ for model in models]
            vars_ = [model.variance_matrix_ for model in models]
    else:
         raise ValueError("Valid environments is 'python'.")
     
    #print(betas)
    betas = np.array(betas)
    vars_ = np.array(vars_)

    # Number of imputations
    m = len(betas)
    
    # Pool the coefficients
    pooled_beta = np.mean(betas, axis=0)

    # Calculate within-imputation and between-imputation variances for coefficients
    within_var_coef = np.mean(vars_, axis=0)
    between_var_coef = np.var(betas, axis=0, ddof=1)
    pooled_var_coef = within_var_coef + (1 + 1/m) * between_var_coef
        
    return pooled_beta, pooled_var_coef, m
    
def pool_baseline_function(models, m, environment):
    if environment=='python': 
        # Pool the baseline survival function
        try:
            baseline_survival_functions = [model.model.baseline_survival_ for model in models]
        except: 
            baseline_survival_functions = [model.baseline_survival_ for model in models]
        # Get common time points
        common_times = set.intersection(*[set(baseline.index) for baseline in baseline_survival_functions])
        aligned_baseline_survival = [baseline.reindex(common_times).interpolate() for baseline in baseline_survival_functions]

        # Stack into a matrix and calculate pooled baseline survival
        baseline_survival_matrix = pd.concat(aligned_baseline_survival, axis=1)

        pooled_baseline_survival = baseline_survival_matrix.mean(axis=1)

        # Calculate within-imputation and between-imputation variances for the baseline survival function
        within_var_baseline = baseline_survival_matrix.var(axis=1, ddof=1)
        between_var_baseline = np.var(baseline_survival_matrix.values, axis=1, ddof=1)
        pooled_var_baseline = within_var_baseline + (1 + 1/m) * between_var_baseline
        
        return pooled_baseline_survival, pooled_var_baseline, common_times
    
    else:
        raise ValueError('valid environments is python')
    
def empty_cox_model_initialization(sufix, files_with_train, time_event_columns, environment):
    if environment=='python': 
        # Create a new CoxPHFitter object

        new_cox_model = CoxPHFitter(penalizer=1000)

        # Fit a dummy dataset to initialize the model structure, after the coefficients and baseline function are replaced
        dummy_data = pd.read_csv('data/'+sufix+"/"+files_with_train[0])

        #print(dummy_data.columns)
        dummy_data.set_index('index', inplace=True)
        #print(dummy_data.columns)

        # Fit the CoxPHFitter model on dummy data to initialize its structure
                    
        new_cox_model.fit(dummy_data, duration_col=time_event_columns[0], event_col= time_event_columns[1])
    
    elif environment=='R': 
        raise ValueError('Not implemented yet')
    else:
        raise ValueError('valid environments are python and R')

    return new_cox_model

def fill_empty_cox_model(new_cox_model, pooled_beta, pooled_var_coef, pooled_baseline_survival,
                         common_times, pooled_var_baseline, environment):
    if environment=='python': 
        # Overwrite the coefficients and baseline survival with the pooled results
        new_cox_model.params_ = pd.Series(pooled_beta, index=new_cox_model.params_.index)
        new_cox_model.variance_matrix_ = pd.DataFrame(pooled_var_coef, index=new_cox_model.params_.index, columns=new_cox_model.params_.index)

        # Assign the pooled baseline survival
        new_cox_model.baseline_survival_ = pd.DataFrame(pooled_baseline_survival, columns=['baseline_survival'])
        new_cox_model.baseline_survival_.index = pd.Index(common_times, name='timeline')

        # Assigning the cumulative baseline hazard
        pooled_cumulative_hazard = -np.log(pooled_baseline_survival)
        new_cox_model.baseline_cumulative_hazard_ = pd.DataFrame(pooled_cumulative_hazard, columns=['baseline_cumulative_hazard'])
        new_cox_model.baseline_cumulative_hazard_.index = pd.Index(common_times, name='timeline')

        # Store the variance of the baseline survival (can be used for calculating confidence intervals)
        new_cox_model.baseline_survival_variance_ = pd.DataFrame(pooled_var_baseline, columns=['baseline_survival_variance'])
        new_cox_model.baseline_survival_variance_.index = pd.Index(common_times, name='timeline')
    elif environment=='R': 
        raise ValueError('Not implemented yet')
    else:
        raise ValueError('valid environments are python and R')
    return new_cox_model

