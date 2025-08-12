import pandas as pd
import miceforest as mf
import os
from src.files import delete_files

class MICE:
    def __init__(self, num_plausible_values, pmm,  t_star, data, phase=None):
        """
        Initialize the Impute class with the given parameters.

        Parameters:
        - num_plausible_values: int, number of plausible values to generate.
        - pmm: bool, whether to use Predictive Mean Matching (PMM) for imputation.
        - save_path: str, path to save the imputed data.
        """
        self.num_plausible_values = num_plausible_values
        self.pmm = pmm
        self.save_path = t_star.path
        self.path_imputed = t_star.path_imputed
        self.imputed_data = []
        self.kernel = None
        self.iterations = 5
        self.data = data
        self.predictors = t_star.predictors
        self.sufix = t_star.file_name_testing.replace('test_','').replace('train_','')
        self.phase = phase
        self.event_column = t_star.time_event[1]
        self.time_column =  t_star.time_event[0]
        self.load = t_star.load
        self.index = None
        
    def create(self, data):
        """
        Create imputed datasets using the specified method.

        Parameters:
        - data: DataFrame, the dataset with missing values to impute.
        - verbose: bool, if True, print the details of the imputation kernel.

        Returns:
        - ImputationKernel, the kernel containing multiple datasets with imputed values.
        """

        data.reset_index(inplace=True)
        
        self.index = data['index']
        
        del data['index']

        data.columns = [col.replace('.0', '') for col in data.columns]
        
        self.kernel = mf.ImputationKernel(
            data[self.predictors],
            random_state=42, 
            num_datasets=self.num_plausible_values,  # Number of datasets with plausible values
            mean_match_candidates=self.pmm if self.pmm else None,  # PMM parameter if PMM is used
        )

        return self.kernel

    def impute(self): 
        self.kernel.mice(self.iterations)
        for i in range(self.num_plausible_values):
            new_data = self.kernel.complete_data(dataset=i)
            self.imputed_data.append(new_data)


    def save(self, delete_old_files):
        """
        Save each imputed dataset to the specified path directory with a unique file name.

        Each dataset is saved as a CSV file with a unique name indicating the dataset index.
        """
        if self.kernel is None:
            print("No imputed data to save. Please run 'create' method first.")
            raise TypeError('No imputed data to save')
        
        # Construct the full directory path
        full_dir_path = os.path.join(self.path_imputed, self.sufix)

        # Ensure the full directory path exists
        os.makedirs(full_dir_path, exist_ok=True)
        
        if delete_old_files:
            delete_files(full_dir_path)
        
        
            
        for i in range(self.num_plausible_values):
            # Retrieve the ith dataset from the kernel
            imputed_df = self.kernel.complete_data(i)
            imputed_df = pd.concat([self.data[[self.time_column, self.event_column]], 
                                    imputed_df], axis=1)

            # Construct the file name and save path
            file_name = self.phase+f"_imputed_dataset_{i+1}.csv"
            full_path = os.path.join(full_dir_path, file_name)
            
            # Save the dataset to a CSV file
            imputed_df.index = self.index
            imputed_df.to_csv(full_path)
            print(f"Dataset {i+1} saved to {full_path}")
            
    def run(self, verbose=True, delete_old_files=False):
        """
        Run the full imputation pipeline: create, impute, save.

        Parameters:
        - data: DataFrame, the dataset with missing values to impute.
        - verbose: bool, if True, print the details of the imputation kernel.
        """
        
        self.create(self.data)
        
        self.save(delete_old_files)
        
