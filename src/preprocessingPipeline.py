import pandas as pd
import yaml 
import numpy as np 
from pathlib import Path

class PreprocessingPipeline:
    def __init__(self, binarization=True, range_check=True, 
                 time_in_dialysis_from_date=True, life_expectancy=False):
        self.binarization = binarization
        self.ranges_checking = range_check
        self.time_in_dialysis_from_date = time_in_dialysis_from_date
        self.tx_date = 'TX_DATE'
        self.dialysis_first_date = 'DIALYSIS_DATE'
        self.life_expectancy = life_expectancy

    def modify_age_eurotransplant(self, value):
        if value < 0: 
            return 0.01
        else: 
            return value
        
    def convert_to_numeric(self, data, t_star, dataset='UNOS'):
        if dataset=='EUROTRANSPLANT': 
            #data['AGE'] = pd.to_numeric(data['AGE'], errors='coerce')
            #data['AGE_DON'] = pd.to_numeric(data['AGE_DON'], errors='coerce') 
            #data['AGE_DON'] = data['AGE_DON'].apply(self.modify_age_eurotransplant)
            #data = data[data[self.dialysis_first_date]>=0] #negative time on dialysis, units of time on dialysis?
            
            #data['CREAT_TRR'] = (data['CREAT_TRR'] - data['CREAT_TRR'].min())/(data['CREAT_TRR'].max() - data['CREAT_TRR'].min())
            #data['AGE'] = (data['AGE'] - data['AGE'].min())/(data['AGE'].max() - data['AGE'].min())
            #data['AGE_DON'] = (data['AGE_DON'] - data['AGE_DON'].min())/(data['AGE_DON'].max() - data['AGE_DON'].min())

            #data['WGT_KG_DON_CALC'] = (data['WGT_KG_DON_CALC'] - data['WGT_KG_DON_CALC'].min())/(data['WGT_KG_DON_CALC'].max() - data['WGT_KG_DON_CALC'].min())
            #data['COD_CAD_DON'] = data['COD_CAD_DON'].replace({'CVA': 1, 'Head trauma': 0,
            #                                                               'Anoxia':0, 'Other':0}).astype("category")
            #data['HIST_HYPERTENS_DON'] = data['HIST_HYPERTENS_DON'].astype("category")
            #data['graft_hcvab'] = data['graft_hcvab'].astype('category')
            #data['graft_dcd'] = data['graft_dcd'].astype('category')
            #data['GSTATUS_KI'] = data['GSTATUS_KI'].astype('bool')
            print('skipping preprocessing in EUROTRANSPLANT')
        else: 
            raise ValueError('We can not manage this data set.')
        return data
     
    def delete_predictor_from_t_star(self, t_star, predictors=[]):
        """
        Deletes specified predictors from the DataFrame `df` and updates the 
        `t_star.predictors` list by removing the same predictors.

        Parameters:
        ----------
        df : pandas.DataFrame
            The DataFrame from which the predictors should be deleted.
        
        t_star : object
            An object that contains a `predictors` attribute (list of predictors).
        
        predictors : list
            A list of predictor names (as strings) to be deleted from the DataFrame 
            and the `t_star.predictors` list.

        Returns:
        -------
        pandas.DataFrame
            The DataFrame after the specified predictors have been removed.

        Raises:
        ------
        ValueError
            If any of the specified predictors do not exist in `df.columns` or `t_star.predictors`.
        """
        
        for predictor in predictors:
            try:
                if predictor in t_star.predictors:
                    t_star.predictors.remove(predictor)
                else:
                    print("Cannot delete {} as it does not exist in t_star.predictors.".format(predictor))
            except Exception as e:
                print(e) 
        return t_star
    
    def delete_predictor_from_date_set(self, df, predictors=[]):
        """
        Deletes specified predictors from the DataFrame `df` and updates the 
        `t_star.predictors` list by removing the same predictors.

        Parameters:
        ----------
        df : pandas.DataFrame
            The DataFrame from which the predictors should be deleted.
        
        t_star : object
            An object that contains a `predictors` attribute (list of predictors).
        
        predictors : list
            A list of predictor names (as strings) to be deleted from the DataFrame 
            and the `t_star.predictors` list.

        Returns:
        -------
        pandas.DataFrame
            The DataFrame after the specified predictors have been removed.

        Raises:
        ------
        ValueError
            If any of the specified predictors do not exist in `df.columns` or `t_star.predictors`.
        """
        
        for predictor in predictors:
            try:
                # Check if the predictor exists in the DataFrame columns
                if predictor in df.columns:
                    df = df.drop(predictor, axis='columns')
                else:
                    raise ValueError(f"Cannot delete '{predictor}' as it does not exist in the DataFrame columns.")
            except Exception as e:
                print(e) 
        return df
                 
    def run(self, t_star):

        df_train = t_star.training_data.copy()
        df_test = t_star.testing_data.copy()     

        df_train.replace('.', np.nan, inplace=True)
        df_test.replace('.', np.nan, inplace=True)
                            
        df_train = self.convert_to_numeric(df_train, t_star, dataset=t_star.data_set)

        df_test = self.convert_to_numeric(df_test, t_star, dataset=t_star.data_set)

        return df_train, df_test
