import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

def save_model(environment,full_dir_path_r, new_cox_model): 
    if environment == "python":
            model_name = full_dir_path_r+'/'+ environment+'_'+'python_combined.plk'
            with open(model_name, 'wb') as f:
                pickle.dump(new_cox_model, f)
                
def save_model_from_list(files_with_train, full_dir_path, model, sufix, predictors=[],
                duration_col =None, event_col=None): 
    os.makedirs(full_dir_path, exist_ok=True)
    delete_files(full_dir_path)
    for file in tqdm(files_with_train, desc="Fitting and saving models..."):
        model.fit(path=sufix+"/"+file, duration_col=duration_col, event_col=event_col, 
                  predictors=predictors)
        number = file.split('_')[-1]
        model.save_model(file_path= full_dir_path+'/'+ model.environment+'_'+number)
        
def get_models(full_dir_path, files_with_train, model, sufix, predictors, 
                duration_col =None, event_col=None):
    models = []
    os.makedirs(full_dir_path, exist_ok=True)
    
    for file in tqdm(files_with_train, desc="Fitting..."):
        model.fit(path=sufix+"/"+file, duration_col=duration_col, event_col=event_col,
                  predictors=predictors)
        models.append(model)
    return models

def get_indexes(path, test_proportion=0.2):
    """
    Splits the index of a DataFrame into training and testing sets.

    Parameters:
    -----------
    path : str
        The file path to a CSV file that contains the data. The file should be readable by pandas' `read_csv` function.

    Returns:
    --------
    train_idx : pandas.Index
        The index values corresponding to the training set.
    test_idx : pandas.Index
        The index values corresponding to the test set.

    Description:
    ------------
    This function reads a CSV file into a pandas DataFrame, extracts its index, and then splits the index into 
    two sets: training and testing. The split is done using an (1-test_proportion)/test_proportion ratio, with 100x(1-test_proportion)% of the index values allocated 
    to the training set and 100xtest_proportion% to the test set. The `random_state` parameter is set to 42 to ensure that the 
    split is reproducible.
    
    The function uses scikit-learn's `train_test_split` function to perform the split, and it returns the training 
    and test indices separately.
    """
    df = pd.read_csv(path)
    df.set_index('index', inplace=True)
    
    train_idx, test_idx = train_test_split(df.index, test_size=test_proportion)  
    
    return train_idx, test_idx

def copy_and_paste_indexes(t_star, sufix, r): 
    
    recovered_path = [f for f in os.listdir('data/') 
                    if os.path.isdir(os.path.join('data/', f)) 
                    and t_star.indexes_from in f][0]
    
    source_path = 'data/'+recovered_path+'/'+str(r)
    target_path = 'data/'+sufix+'/'+str(r)
    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    path_to_repetition_train = 'data/'+sufix+'/' + str(r)+ '/'
    path_to_repetition_test = 'data/'+sufix+'/' + str(r)+ '/after_t_star/'
    train_idx = pd.read_csv(path_to_repetition_train+'train_idx.csv')['index'].values 
    validation_idx = pd.read_csv(path_to_repetition_train+'test_idx.csv')['index'].values
    test_idx = pd.read_csv(path_to_repetition_test+'test_idx.csv')['index'].values
    unuseful_idx = None
    
    return train_idx, validation_idx, test_idx,  unuseful_idx

def save_indexes(train_idx, test_idx, path):
    """
    Saves the training and testing indices to separate CSV files.

    Parameters:
    -----------
    train_idx : array-like
        The indices for the training data to be saved.
        
    test_idx : array-like
        The indices for the testing data to be saved.
        
    path : str
        The directory path where the CSV files will be saved.

    Description:
    ------------
    This function takes two sets of indices (training and testing) and saves them as CSV files. 
    The training indices are saved as 'train_idx.csv' and the testing indices are saved as 'test_idx.csv'
    in the specified directory.
    """
    os.makedirs(path, exist_ok=True)
    delete_files(path)
    
    pd.DataFrame(train_idx).to_csv(path+'/'+'train_idx.csv')
    pd.DataFrame(test_idx).to_csv(path+'/'+'test_idx.csv')

def set_directories(sufix, r):
    """
    Creates and organizes directories for models, data, and predictions based on the provided suffix and iteration value.

    Parameters:
    -----------
    sufix : str
        A suffix to use for creating the directories (e.g., experiment name or model type).
        
    r : int
        An integer representing an iteration or version (used as part of the directory structure).

    Returns:
    --------
    full_dir_path_pred : str
        The full path to the directory for storing predictions (based on `sufix` and `r`).
        
    full_dir_path_data : str
        The full path to the directory for storing data (based on `sufix` and `r`).
        
    full_dir_path_r : str
        The full path to the directory for storing models (based on `sufix` and `r`).

    Description:
    ------------
    This function creates directories to store models, data, and predictions based on a given `sufix` and iteration `r`. 
    The directory structure is as follows:
    
    - Models are saved in `'models/{sufix}/{r}'`.
    - Data is saved in `'data/{sufix}/{r}'`.
    - Predictions are saved in `'predictions/{sufix}/{r}'`.
    
    The function creates the necessary directories if they do not already exist, ensuring that the structure is set up for saving model outputs, related data, and prediction results.
    
    The directories are created with `exist_ok=True`, meaning the function will not raise an error if the directories already exist.
    
    The function returns the paths to the predictions directory, data directory, and models directory, respectively.
    """
    full_dir_path = os.path.join('models/', sufix+'/')
    full_dir_path_r = os.path.join(full_dir_path, str(r))
    os.makedirs(full_dir_path_r, exist_ok=True)
    delete_files(full_dir_path_r)

    full_dir_path_data = os.path.join('data/'+sufix+'/', str(r)) 
    os.makedirs(full_dir_path_data, exist_ok=True)
    delete_files(full_dir_path_data)
    
    full_dir_path_predictions = os.path.join('predictions', sufix) 
    full_dir_path_pred = os.path.join(full_dir_path_predictions, str(r)) 
    os.makedirs(full_dir_path_pred, exist_ok=True)
    delete_files(full_dir_path_pred)
    

    return full_dir_path_pred, full_dir_path_data, full_dir_path_r
     
def delete_files(full_dir_path):
    if os.path.exists(full_dir_path):
        for filename in os.listdir(full_dir_path):
            file_path = os.path.join(full_dir_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'Directory {full_dir_path} does not exist.')