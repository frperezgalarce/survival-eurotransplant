from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd 
from dateutil.relativedelta import relativedelta
from pathlib import Path
import src.utils as ut
import warnings
import yaml 

warnings.filterwarnings("ignore")

class TStar:
    def __init__(self, t_star_date="31-12-2009", txpt=5, ept=7, txpp=1, epp=10, 
                 load=False, training_data_dir=None, testing_data_dir=None, 
                 indexes_from_ID=None, dataset = 'UNOS'):
        ''' training_data and testing_data are used to load directly the data sets without
        filtering'''
        self.indexes_from = indexes_from_ID
        self.t_star_date = datetime.strptime(t_star_date, "%d-%m-%Y")  # Date where we can predict
        self.txpt = txpt  # Transplant period for training (years)
        self.ept = ept  # Event period for training (years)
        self.txpp = txpp  # Transplant period for predicting (years)
        self.epp = epp  # Event period for predicting (years)
        self.base_year = pd.to_datetime(self.t_star_date)- relativedelta(years=self.ept) + relativedelta(days=1)
        self.id = pd.read_csv('id.csv')
        self.dir_name = self._set_dir_name()
        self.file_name_testing = "test_" + self.dir_name
        self.file_name_training = "train_" + self.dir_name
        self.path, self.path_imputed, self.path_models, self.path_predictions  = self.get_path_to_data()
        self.time_event = ['GTIME_KI', 'GSTATUS_KI']
        self.training_data = None  # Data set for training, initialized as None
        self.testing_data = None  # Data set for testing, initialized as None
        self.load = load  # Boolean variable indicating if the data is loaded or created
        self.columns = None
        self.predictors = None
        self.training_data_dir = training_data_dir
        self.testing_data_dir = testing_data_dir
        self.check_values()
        self.data_set = dataset
       
    def get_path_to_data(self):
        with open(Path(__file__).parent.parent/'paths.yaml', 'r') as file:
            paths = yaml.safe_load(file)
        return paths['PATH_DATA'], paths['PATH_IMPUTED'], paths['PATH_MODELS'], paths['PATH_PREDICTION']    
         
    def check_values(self):
        if self.txpt>self.ept: 
            raise('There are inconsistent arguments: self.txpt>self.ept')
        if self.txpp>self.epp: 
            raise('There are inconsistent arguments: self.txpp>self.epp')
        if self.base_year<datetime.strptime("01-01-1985", "%d-%m-%Y"):
            raise ValueError("The baseline year must be after 01-01-1985.")
        
    def _set_dir_name(self): 
        self.id['value'] = self.id['value'] + 1
        self.id.to_csv('id.csv', index=False)
        dir_name = (("t_"+str(self.t_star_date)+"_txpt_"+str(self.txpt)
                    +"_ept_"+str(self.ept)+"_txpp_"+str(self.txpp)
                    +"_epp_"+str(self.epp)).replace(':','').replace(' ','')
                    +'_ID_'+str(self.id.value.loc[0]))
        return dir_name
     
    def _load_training_data(self):
        """Method to load training data."""
        file_name = self.file_name_training+".csv"
        data = pd.read_csv(Path(__file__).parent.parent/"data"/file_name, index_col=False)
        return data

    def _load_testing_data(self):
        """Method to load testing data."""
        file_name = self.file_name_testing+".csv"
        data = pd.read_csv(Path(__file__).parent.parent/"data"/file_name, index_col=False)
        return data
        
    def set_predictors(self, predictors, columns):
        self.columns = columns
        self.predictors = predictors
        
    def get_data(self, model_type='cphm', time_variable_name='GTIME_KI', status_variable_name= 'GSTATUS_KI'):
        if self.testing_data_dir is None or self.training_data_dir is None:
            if self.data_set == 'EUROTRANSPLANT': 
                if self.load: 
                    self.testing_data = self._load_testing_data()
                    self.training_data = self._load_training_data()
                else: 
                    self._create_data_sets(model_type=model_type, time_variable_name=time_variable_name, 
                                           status_variable_name=status_variable_name)
            else: 
                raise ValueError('This dataset can not be hanlded by this project. Please contact the developing team')
        else:
            self.testing_data = pd.read_csv(self.testing_data_dir, index_col=0) 
            self.training_data = pd.read_csv(self.training_data_dir, index_col=0)

    def merge(self, kidpan_df, deceased_df):
        
        kidpan_df.reset_index(inplace=True)
        df = kidpan_df.merge(deceased_df, how='left', on='DONOR_ID')
        df = df.set_index('index')
        del kidpan_df, deceased_df
        df = df[self.columns]
        
        return df

    def load_and_merge(self): 
        
        if self.data_set == 'UNOS':
            raise ValueError()
        elif self.data_set=='EUROTRANSPLANT': 
                       
            #df = pd.read_csv('C:/Users/HP/Desktop/EUROTRANSPLANT/example_dataset.csv')
            df = pd.read_csv('/home/franciscoperez/Documents/GitHub/validation-franework-eurotransplant/data/formatted_dataset.csv')
            
            #print(df.shape)
            #columns_to_delete = ['time_to_death', 'death', 'time_to_retx_or_death', 
            #                     'retx_or_death', 'time_to_rereg', 'rereg',
            #                    'time_to_acgl', 'acgl']
       
            #df.drop(columns_to_delete, axis=1, inplace=True)
            
            #replace_column_names = {'time_to_dcgl': 'GTIME_KI', 
            #                        'dcgl': 'GSTATUS_KI',
            #                        'year_txp': 'TX_DATE', 'pseudoid': 'index',
            #                        'donor_age': 'AGE_DON', 
            #                        'donor_last_creat': 'CREAT_TRR', 
            #                        'donor_weight':'WGT_KG_DON_CALC', 
            #                        'd_height': 'HGT_CM_DON_CALC',
            #                        'death_cause_group':'COD_CAD_DON', 
            #                        'time_on_dial':'DIALYSIS_DATE', 
            #                       'wfmr_r_age': 'AGE', 
            #                        'donor_hypertension': 'HIST_HYPERTENS_DON'}
                    
            #print('time on dial', df['time_on_dial'])
            #print('death_cause_group', df['death_cause_group'].unique())
            
            #df.rename(columns=replace_column_names, inplace=True)
            
            df['TX_DATE'] = pd.to_datetime(df['year_txp'].astype(str), format='%Y')
            #if df.GTIME_KI.min() < 0: 
            #    print('There are negative values for time.')
                
            #df = df[df.GTIME_KI!=-1920]
            #df.GTIME_KI = df.GTIME_KI.apply(self.euro_negaative_values)
            #df.AGE = df.AGE.apply(self.euro_negaative_values)
            
            print(df.shape)
            df = self.set_time_variables(df, time_variable_name='time_to_dcgl')
        else: 
            raise ValueError('We are not able to handle this dataset')
        
        
        print(df.shape, 'after set_time')
        return df
    def euro_negaative_values(self, value): 
        if value < 0: 
            return 1
        else: 
            return value
    
    def set_time_variables(self, df, time_variable_name = 'GTIME_KI'):
        """
        This function sets the time variables for the DataFrame by converting 'TX_DATE' to datetime
        and calculating a new 'Tag_date' by adding the number of days specified in 'T' to 'TX_DATE'.

        Parameters:
        df (DataFrame): The input DataFrame containing 'TX_DATE' and 'T' columns.
        verbose (bool): If True, prints the DataFrame with the new 'Tag_date' column.

        Returns:
        DataFrame: The DataFrame with the new 'Tag_date' column.
        """
        df['TX_DATE'] = pd.to_datetime(df['TX_DATE'])

        # If 'T' represents a number of days to shift, convert 'T' to an integer type if it's not already
        df[time_variable_name] = df[time_variable_name].astype(int)

        # Display the resulting DataFrame with the new 'Tag_date' column

        df['Tag_date'] = df['TX_DATE'] + pd.to_timedelta(df[time_variable_name], unit='D')
        
        return df
    
    
    
    def add_predictor(self, variable_name): 
        self.predictors.append(variable_name)
        
    def update_predictors(self):
        
        with open(Path(__file__).parent.parent/'config_binaries_unos.yaml', 'r') as file:
            config_binaries = yaml.safe_load(file)

        for element in config_binaries['config_binaries'].items(): 
            variable, threshold_info = element
            
            variable = variable.replace('_99', '')
            type = threshold_info['type']
            threshold_type = threshold_info['threshold_type']
            value = threshold_info['threshold_value']  
            variable_name = variable+threshold_type+str(value)
            
            if type == 'continuous':
                self.predictors.append(variable_name)
            
            elif type == 'aggregated-categorical':
                for v in value:
                    variable_name = variable+threshold_type+str(v)
                    variable_name = variable_name.replace('None','_') 
                    self.predictors.append(variable_name)
            
            elif type == 'categorical': 
                variable_name = variable_name.replace('None','_')
                self.predictors.append(variable_name)
            
            else: 
                raise('Error')
            
            if threshold_info['remove_original']: 
                try:
                    self.predictors.remove(variable)
                except Exception as error: 
                    if self.data_set=='UNOS':
                        raise ValueError(error)
                    elif self.data_set=='EUROTRANSPLANT':
                        print('{} was not deleted since {}'.format(variable, error) ) 

    def _create_data_sets(self, model_type='cphm', status_variable_name = 'GSTATUS_KI', 
                          time_variable_name='GTIME_KI'):
        """
        This function creates training and testing datasets based on given parameters.

        Parameters:
        verbose (bool): If True, prints detailed information about the periods and dataset sizes.

        """

        # Calculate the final dates for the training, event, and prediction periods
        t_final_training_period = pd.to_datetime(self.base_year)+ relativedelta(years=self.txpt) - relativedelta(days=1)
        t_final_event_period = pd.to_datetime(self.base_year)+ relativedelta(years=self.ept) - relativedelta(days=1)
        t_truncate_event = pd.to_datetime(self.base_year)+ relativedelta(years=self.ept - 1) - relativedelta(days=1)
        t_final_prediction_period = pd.to_datetime(self.t_star_date)+ relativedelta(years=self.epp)        
        t_final_prediction_tx_period = pd.to_datetime(self.t_star_date)+ relativedelta(years=self.txpp) 
        
        if t_final_prediction_period>datetime.strptime("31-12-2023", "%d-%m-%Y"):
            raise ValueError("The baseline year must be after 31-12-2023.")
        if model_type not in ['competing-risks', 'cphm']:
            raise ValueError('This model is not implemented in this pipeline.')
        
        print('t_b: ', self.base_year)
        print('t_final_training_period: ', t_final_training_period)
        print('t_final_event_period: ', t_final_event_period)
        print('t_final_prediction_period: ', t_final_prediction_period)
        print('t_final_prediction_tx_period: ', t_final_prediction_tx_period)
        
        df_raw = self.load_and_merge()
        
        if model_type == 'competing-risks':
            df_raw['GSTATUS_KI'] = df_raw['GSTATUS_KI'] + (~df_raw['COMPOSITE_DEATH_DATE'].isna() & (df_raw['GSTATUS_KI']!=1))*2


        print(self.base_year)
        print(df_raw['TX_DATE'])
        df = df_raw[(df_raw['TX_DATE']>=self.base_year)]
        
        print(df.shape, 'after filtering by base year')
        
        df_testing = df[(df['TX_DATE']>t_final_event_period) & (df['TX_DATE']<=t_final_prediction_tx_period)]
        df_testing[status_variable_name] = df_testing[status_variable_name].astype('bool')
        print(df.shape, 'after filtering by testing window')
        
        df_training_without_editing = df[(df['TX_DATE']<=t_final_training_period) &
                                         (df['Tag_date']<=t_truncate_event)]
        print(df.shape, 'after filtering by testing window')
        df_training_to_edit = df[~df.index.isin(df_training_without_editing.index)]
        
        if self.data_set=='UNOS':
            df_training_to_edit = self.shift_time_status(df_training_to_edit, t_final_training_period, 
                                                         t_final_event_period, t_truncate_event)
        
        df_training = pd.concat([df_training_without_editing, df_training_to_edit], axis=0)
        
        df_training[status_variable_name] = df_training[status_variable_name].astype('bool')

        #TODO: ver si esto debe ser para un flag 0 o 1.
        status_to_select_data = False
        if (df_training[df_training[status_variable_name]==status_to_select_data][time_variable_name].max() < 
                       df_testing[df_testing[status_variable_name]==status_to_select_data][time_variable_name].max()):
            print(df_training[df_training[status_variable_name]==status_to_select_data][time_variable_name].max())
            print(df_testing[df_testing[status_variable_name]==status_to_select_data][time_variable_name].max())

            # Ensure filtering conditions are correctly applied
            to_insert = df_raw[df_raw[time_variable_name]==status_to_select_data][
                (df_raw['TX_DATE'] <= t_final_training_period) &
                (df_raw['Tag_date'] <= t_truncate_event) &
                (df_raw[time_variable_name] >= df_testing[time_variable_name].max())
            ]

            # Sort and safely select the first row
            to_insert = to_insert.sort_values(time_variable_name, ascending=True)

            if not to_insert.empty:
                to_insert = to_insert.iloc[[0]]  # Select as DataFrame

                df_training = pd.concat([df_training, to_insert])

                print(df_training.shape)
            else:
                print("No rows found to insert. Then, it was inserted one row from the testing set.")
                to_insert_from_testing = df_testing[df_testing[status_variable_name]==status_to_select_data].sort_values([time_variable_name],
                                                                                                              ascending=True)
                to_insert_from_testing = to_insert_from_testing.iloc[[0]]
                df_training = pd.concat([df_training, to_insert_from_testing])

                print(df_training.shape)
                
        self.training_data = df_training
        df_training.to_csv('data/'+self.file_name_training+".csv")
        self.testing_data = df_testing
        df_testing.to_csv('data/'+self.file_name_testing+".csv") 
    
    def visualize_t_star(self):
        """Method to visualize the TStar timeline."""
        # Calculate important dates
        self.base_year = self.t_star_date - timedelta(days=365 * self.ept)
        txpt_end_date = self.base_year + timedelta(days=365 * self.txpt)
        ept_end_date = self.base_year + timedelta(days=365 * self.ept)
        txpp_end_date = self.t_star_date + timedelta(days=365 * self.txpp)
        epp_end_date = self.t_star_date + timedelta(days=365 * self.epp)

        # Plotting the timeline
        plt.figure(figsize=(10, 2))

        plt.plot([self.base_year, txpt_end_date], [1, 1], color="blue", label="Training Transplant Period")
        plt.plot([self.base_year, ept_end_date], [0, 0], color="green", label="Training Event Period")
        plt.plot([self.t_star_date, txpp_end_date], [-1, -1], color="orange", label="Predicting Transplant Period")
        plt.plot([self.t_star_date, epp_end_date], [-2, -2], color="red", label="Predicting Event Period")

        plt.scatter([self.t_star_date], [1], color="black", label="t_star_date", zorder=5)
        
        plt.yticks([])
        plt.xlabel("Date")
        plt.title("TStar Timeline")
        plt.legend()
        plt.grid(True)
        plt.show()

    def shift_time_status(self, df, t_final_training_period, t_final_event_period, t_truncate_event):
        
        df = df.copy()  
        
        if self.data_set=='UNOS':
            df['GSTATUS_KI'] = df['GSTATUS_KI'] + (~df['COMPOSITE_DEATH_DATE'].isna() & (df['GSTATUS_KI']!=1))*2
        
        df.reset_index(inplace=True)
        
        # all status with T higher than t* are defined censored
        df_training_to_edit_1 = df[(df['TX_DATE']<=t_final_training_period) & 
                                (df['Tag_date']>t_final_event_period)] 
        
        df_training_to_edit_1['GSTATUS_KI'] = 0 # Mark as censored
        df_training_to_edit_1['GTIME_KI'] = (t_final_event_period - 
                                             df_training_to_edit_1['TX_DATE']).dt.days
                
        # censored between t*-1 year and t* year are defined censored at t*
        df_training_to_edit_2 = df[(df['TX_DATE']<=t_final_training_period) & 
                                   (df['Tag_date']>t_truncate_event) &  
                                   (df['Tag_date']<=t_final_event_period) & 
                                   (df['GSTATUS_KI']==0) ]
         
        df_training_to_edit_2['GSTATUS_KI'] = 0
        df_training_to_edit_2['GTIME_KI'] = (t_final_event_period - 
                                             df_training_to_edit_2['TX_DATE']).dt.days
                
        # graft failure event between t*-1 year and t* year are not modified
        df_training_to_edit_3 = df[(df['TX_DATE']<=t_final_training_period) & 
                                   (df['Tag_date']>t_truncate_event) &  
                                   (df['Tag_date']<=t_final_event_period) & 
                                   (df['GSTATUS_KI']==1)] 
                
        # Case when a patient dies between t*-1 year and t* year T and T are not modified
        df_training_to_edit_4 = df[(df['TX_DATE']<=t_final_training_period) & 
                                   (df['Tag_date']>t_truncate_event) &  
                                   (df['Tag_date']<=t_final_event_period) & 
                                   (df['GSTATUS_KI']==2)]
        
        df_training_to_edit_4['GSTATUS_KI'] = 0
        
        df_training_to_edit = pd.concat([df_training_to_edit_1, df_training_to_edit_2, 
                                         df_training_to_edit_3, df_training_to_edit_4], 
                                        axis=0)
        
        df_training_to_edit.sort_values('index', inplace=True)
        df_training_to_edit.set_index('index', inplace=True)
        return df_training_to_edit
    