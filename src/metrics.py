import pandas as pd
import numpy as np
import os
from sksurv.metrics import brier_score, concordance_index_censored, cumulative_dynamic_auc
from datetime import datetime
import seaborn as sns
from sksurv.nonparametric import CensoringDistributionEstimator
import copy

class Metrics:
    def __init__(self,t_star_date, txpt, ept, txpp, epp, imputed=True):
        self.t_star_date = datetime.strptime(t_star_date, "%d-%m-%Y")  # Date where we can predict
        self.txpt = txpt  # Transplant period for training (years)
        self.ept = ept  # Event period for training (years)
        self.txpp = txpp  # Transplant period for predicting (years)
        self.epp = epp
        self.path_name = ("t_"+str(self.t_star_date)+"_txpt_"+str(self.txpt)
                                  +"_ept_"+str(self.ept)+"_txpp_"+str(self.txpp)
                                  +"_epp_"+str(self.epp)).replace(':','').replace(' ','')
        self.imputed = imputed
    
    def brier_score(self, subgroup_df = None, days = int(365.25*5), predictions = None, inputs = None, **kwargs):
        return self._iterate_metric_over_files(self._calculate_brier_score, 'Brier_score', subgroup_df = subgroup_df, days=days, predictions = predictions, inputs = inputs, **kwargs)
    
    def auc(self, subgroup_df = None, days = 365.25*5, predictions = None, inputs = None, **kwargs):
        return self._iterate_metric_over_files(self._calculate_auc, 'AUC', subgroup_df = subgroup_df, days=days, predictions = predictions, inputs = inputs, **kwargs)
    
    def mean_error(self, subgroup_df = None, days = 365.25*5, predictions = None, inputs = None, **kwargs):
        return self._iterate_metric_over_files(self._calculate_mean_error, 'Mean_error', subgroup_df = subgroup_df, days=days, predictions = predictions, inputs = inputs, **kwargs)
    
    def mean_squared_error(self, subgroup_df = None, days = 365.25*5, predictions = None, inputs = None, **kwargs):
        return self._iterate_metric_over_files(self._calculate_mean_squared_error, 'Mean_error', subgroup_df = subgroup_df, days=days, predictions = predictions, inputs = inputs, **kwargs)
    
    def concordance_index(self, subgroup_df = None, surv_time = None,  predictions = None, inputs = None, **kwargs):
        return self._iterate_metric_over_files(self._calculate_concordance_index, 'Concordance_index', subgroup_df = subgroup_df, surv_time=surv_time, predictions = predictions, inputs = inputs, **kwargs)
    
    def subgroup_c_index(self, subgroup_df = None, surv_time=None, inputs=None, predictions=None, duration_col='GTIME_KI', event_col='GSTATUS_KI'):
        results = {}
        idx = 0
        for (k, input_df) in inputs.items():
            input_df = input_df.copy()
            input_df['group'] = input_df.index.map(subgroup_df)
            groups = input_df['group'].unique()
            for surv_df in predictions[k]:
                for group in groups:
                    #print(group)
                    if input_df[input_df['group'] == group].empty:
                        results[idx] = {
                            'input_id':k, 
                            'Group-C-index': np.nan, 
                            'group': group, 
                            'group_size': np.nan
                                }
                    else:    
                        results[idx] = {
                            'input_id':k, 
                            ' Group-C-index': self._calculate_subgroup_c_index( surv_df, input_df, duration_col=duration_col, event_col=event_col, surv_time = surv_time, group_col = 'group', group_val = group), 
                            'group': group, 
                            'group_size': input_df[input_df['group']==group].shape[0] }
                    idx +=1
        df = pd.DataFrame.from_dict(results, orient='index')
        return df

   
    
    
    
    def within_c_index(self, subgroup_df = None, surv_time=None, inputs=None, predictions=None, duration_col='GTIME_KI', event_col='GSTATUS_KI'):
        results = {}
        idx = 0
        for (k, input_df) in inputs.items():
            input_df = input_df.copy()
            input_df['group'] = input_df.index.map(subgroup_df)
            groups = input_df['group'].unique()
            for surv_df in predictions[k]:
                for group in groups:
                    if input_df[input_df['group'] == group].empty:
                        results[idx] = {
                            'input_id':k, 
                            'Group-C-index': np.nan, 
                            'group': group, 
                            'group_size': np.nan
                                }
                    else:    
                        results[idx] = {
                            'input_id':k, 
                            ' Group-C-index': self._calculate_within_c_index( surv_df, input_df, duration_col=duration_col, event_col=event_col, surv_time = surv_time, group_col = 'group', group_val = group), 
                            'group': group, 
                            'group_size': input_df[input_df['group']==group].shape[0] }
                    idx +=1
        df = pd.DataFrame.from_dict(results, orient='index')
        return df

    def between_c_index(self, subgroup_df = None, surv_time=None, inputs=None, predictions=None, duration_col='GTIME_KI', event_col='GSTATUS_KI'):
        results = {}
        idx = 0
        for (k, input_df) in inputs.items():
            input_df = input_df.copy()
            input_df['group'] = input_df.index.map(subgroup_df)
            groups = input_df['group'].unique()
            for surv_df in predictions[k]:
                for group in groups:
                    if input_df[input_df['group'] == group].empty:
                        results[idx] = {
                            'input_id':k, 
                            'Group-C-index': np.nan, 
                            'group': group, 
                            'group_size': np.nan
                                }
                    else:    
                        results[idx] = {
                            'input_id':k, 
                            ' Group-C-index': self._calculate_between_c_index( surv_df, input_df, duration_col=duration_col, event_col=event_col, surv_time = surv_time, group_col = 'group', group_val = group), 
                            'group': group, 
                            'group_size': input_df[input_df['group']==group].shape[0] }
                    idx +=1
        df = pd.DataFrame.from_dict(results, orient='index')
        return df   
    def expected_c_index(self, input_df, duration_col='GTIME_KI', event_col='GSTATUS_KI', hazard_col = None):
        max_surv_time = input_df[duration_col].max()

        number_patients = input_df.shape[0]
        event_indicator = input_df[event_col].values 
        event_time      = input_df[duration_col].values 
        estimated_risk  = input_df[hazard_col].values 
        #group_indicator = (input_df[group_col] == group_val).values

        sort_mask = np.argsort(event_time)
        event_time = event_time[sort_mask]
        event_indicator = event_indicator[sort_mask]
        estimated_risk = estimated_risk[sort_mask]
        #group_indicator = group_indicator[sort_mask]
        
        idx = 0        
        current_time = event_time[idx]
        total_pairs   = 0
        correct_pairs = 0
        tied_pairs    = 0
        while current_time < max_surv_time and idx < number_patients-1:
            while (event_indicator[idx] == False) and idx < number_patients-1:
                idx += 1
            if idx == number_patients-1:
                break
            total_pairs += number_patients - idx-1
            current_risk = estimated_risk[idx]
            #correct_pairs += np.sum(estimated_risk[idx+1:] < current_risk )
            correct_pairs += np.sum(current_risk/(current_risk + estimated_risk[idx+1:] ))
            #correct_pairs += np.sum(estimated_risk[idx+1:]/(estimated_risk[idx+1:] + current_risk))
            #correct_pairs += np.sum(np.exp(current_risk)/ (np.exp(current_risk) + np.exp(estimated_risk[idx+1:])))
            #tied_pairs += np.sum(estimated_risk[idx+1:] == current_risk )
            idx +=1
        #print("Total pairs {}, correct_pairs {}, tied_pairs {}".format(total_pairs,correct_pairs,tied_pairs))
        return (correct_pairs)/total_pairs    

    
    def _iterate_metric_over_files(self, metric, _name, subgroup_df = None,  predictions = None, inputs = None, visualize = False, **kwargs):
        results = {}
        idx = 0
        if subgroup_df is None:
            groups = ['0']
        else:
            groups = list(subgroup_df.unique())
        
        if predictions is None:
            predictions = self._get_predictions()
        
        if inputs is None:
            inputs = self._get_inputs()

        #files = os.listdir('data/'+self.path_name) if self.imputed else [ x for x  in os.listdir('data') if self.path_name in x]

        for (k, input_df) in inputs.items():
            if subgroup_df is not None:
                    input_dfs = {g: (input_df[input_df.index
                                                      .isin(list(subgroup_df[subgroup_df==g]
                                                      .index))] 
                                    )
                                 for g in groups}
            
            for surv_df in predictions[k]:
                print('surv_df: ', surv_df)
                for group in groups:
                            #print(group)
                    if subgroup_df is None:
                        filt_input_df = input_df
                    else:
                        filt_input_df  = input_dfs[group]
                    if filt_input_df.shape[0]<5:
                        results[idx] = {
                            'input_id':k, 
                            _name: np.nan, 
                            'group': group, 
                            'group_size': filt_input_df.shape[0]
                            }
                    else:
                        results[idx] = {
                            'input_id':k, 
                            _name: metric(surv_df, filt_input_df, **kwargs),
                            'group': group, 
                            'group_size': filt_input_df.shape[0]}
                    idx += 1
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv('results/'+self.path_name + '.csv')
        if visualize:
            fig = sns.displot(df[_name])
            fig.savefig('results/fig/' + _name +'_' +self.path_name + '.png')
        return df


    def _build_median_func(self, surv_df, surv_time):
        __surv_time = surv_time
        if surv_time is not None:
            __surv_time = np.min([i for i in surv_df.index if i >= surv_time])
        def find_median(pt):
            pt = str(pt)
            if pt not in surv_df.columns:
                return np.nan
            col = surv_df[pt]
            if surv_time is None:
                median_survival = list(col[col <= 0.5].index)
                if len(median_survival) == 0:
                    return np.max(col.index)
                else:
                    return np.min(median_survival)
            return col[__surv_time]
        
        return find_median
    
    def _get_inputs(self):
        files = os.listdir('data/'+self.path_name) if self.imputed else [ x for x  in os.listdir('data') if self.path_name in x]

        input_path = lambda file: 'data/' + (self.path_name+'/'+file if self.imputed else file)

        files = [x for x in files if 'test' in x.lower()]

        keys = [x.split('_')[-1].replace('.csv', '') for x in files] if self.imputed else range(len(files))
        return {k: pd.read_csv(input_path(file)) for (k,file) in zip(keys, files) }
    
    def _get_predictions(self):

        predictions = {}

        files = os.listdir('predictions/'+self.path_name) if self.imputed else [ x for x  in os.listdir('predictions') if self.path_name in x]
        
        files = [x for x in files if '.csv' in x]

        if self.imputed:
            files = [x for x in files if 'imputed_' in x ]
            for file in files:
                impute_id = file.split('_')[-1].replace('.csv', '')
                if impute_id not in predictions:
                    predictions[impute_id] = []
                surv_df = pd.read_csv('predictions/'+self.path_name+'/'+file, index_col=0)
                predictions[impute_id].append(surv_df)
            return predictions
        
        keys = range(len(files))
        return {k: [pd.read_csv('predictions/'+self.path_name+'/'+file, index_col=0)] for (k,file) in zip(keys, files)}
        
    def _calculate_concordance_index(self, surv_df, input_df, duration_col='GTIME_KI', event_col='GSTATUS_KI', surv_time = None ):
        """calculates the concordance index of a prediction using -median or -probability of survival at time t as the estimated risk """
        
        find_median = self._build_median_func(surv_df, surv_time)
        input_df['Med_Surv'] = input_df.index.map(find_median)
        event_indicator = input_df[event_col]
        event_time      = input_df[duration_col]
        estimated_risk  = -input_df['Med_Surv']

        cindex, concordant, discordant, tied_risk, tied_time = concordance_index_censored(event_indicator, event_time, estimated_risk)

        return cindex
    
    # def _calculate_subgroup_c_index(self, surv_df, input_df, duration_col='GTIME_KI', event_col='GSTATUS_KI', surv_time = None, group_col = None, group_val = None ):

    #     find_median = self._build_median_func(surv_df, surv_time)
    #     input_df['Med_Surv'] = input_df.index.map(find_median)
    #     max_surv_time = np.max(surv_df.index)
    #     number_patients = input_df.shape[0]
    #     event_indicator = input_df[event_col].values 
    #     event_time      = input_df[duration_col].values + np.random.uniform(-0.03, 0.03, size = number_patients)*(input_df[duration_col].values < max_surv_time)
    #     # estimated_risk  = -input_df['Med_Surv'].values + np.random.uniform(-0.03, 0.03, size = number_patients)*(input_df['Med_Surv'].values < max_surv_time)
    #     estimated_risk  = -input_df['Med_Surv'].values 
    #     group_indicator = (input_df[group_col] == group_val).values

    #     sort_mask = np.argsort(event_time)
    #     event_time = event_time[sort_mask]
    #     event_indicator = event_indicator[sort_mask]
    #     estimated_risk = estimated_risk[sort_mask]
    #     group_indicator = group_indicator[sort_mask]
        
    #     idx = 0        
    #     current_time = event_time[idx]
    #     total_pairs   = 0
    #     correct_pairs = 0
    #     tied_pairs    = 0
    #     while current_time < max_surv_time and idx < number_patients-1:
    #         while (group_indicator[idx] == False or event_indicator[idx] == False) and idx < number_patients-1:
    #             idx += 1
    #         if idx == number_patients-1:
    #             break
    #         total_pairs += number_patients - idx-1
    #         current_risk = estimated_risk[idx]
    #         correct_pairs += np.sum(estimated_risk[idx+1:] < current_risk )
    #         tied_pairs += np.sum(estimated_risk[idx+1:] == current_risk )
    #         idx +=1
    #     #print("Total pairs {}, correct_pairs {}, tied_pairs {}".format(total_pairs,correct_pairs,tied_pairs))
    #     return (correct_pairs + 0.5*tied_pairs)/total_pairs
    
    def _calculate_subgroup_c_index(self, surv_df, input_df, duration_col='GTIME_KI', event_col='GSTATUS_KI', surv_time = None, group_col = None, group_val = None ):

        find_median = self._build_median_func(surv_df, surv_time)
        input_df['Med_Surv'] = input_df.index.map(find_median)
        max_surv_time = np.max(input_df[duration_col])
        number_patients = input_df.shape[0]
        event_indicator = input_df[event_col].values 
        event_time      = input_df[duration_col].values + np.random.uniform(-0.03, 0.03, size = number_patients)*(input_df[duration_col].values < max_surv_time)
        # estimated_risk  = -input_df['Med_Surv'].values + np.random.uniform(-0.03, 0.03, size = number_patients)*(input_df['Med_Surv'].values < max_surv_time)
        estimated_risk  = -input_df['Med_Surv'].values 
        group_indicator = (input_df[group_col] == group_val).values

        sort_mask = np.argsort(event_time)
        event_time = event_time[sort_mask]
        event_indicator = event_indicator[sort_mask]
        estimated_risk = estimated_risk[sort_mask]
        group_indicator = group_indicator[sort_mask]
        
        idx = 0        
        current_time = event_time[idx]
        total_pairs   = 0
        correct_pairs = 0
        tied_pairs    = 0
        while current_time < max_surv_time and idx < number_patients-1:
            while (event_indicator[idx] == False) and idx < number_patients-1:
                idx += 1
            if idx == number_patients-1:
                break
            current_risk = estimated_risk[idx]
            current_is_group = group_indicator[idx]
            if current_is_group:
                total_pairs += number_patients - idx-1 + np.sum(group_indicator[idx+1:])
                correct_pairs += np.sum(estimated_risk[idx+1:] < current_risk) + np.sum((estimated_risk[idx+1:] < current_risk) * group_indicator[idx+1:])   
                tied_pairs += np.sum(estimated_risk[idx+1:] == current_risk) + np.sum((estimated_risk[idx+1:] == current_risk) * group_indicator[idx+1:] )
            else:
                total_pairs += np.sum(group_indicator[idx+1:])
                correct_pairs += np.sum((estimated_risk[idx+1:] < current_risk) * group_indicator[idx+1:] )
                tied_pairs += np.sum((estimated_risk[idx+1:] == current_risk) *  group_indicator[idx+1:])
            idx +=1
        #print("Total pairs {}, correct_pairs {}, tied_pairs {}".format(total_pairs,correct_pairs,tied_pairs))
        return (correct_pairs + 0.5*tied_pairs)/total_pairs
    

    def _calculate_within_c_index(self, surv_df, input_df, duration_col='GTIME_KI', event_col='GSTATUS_KI', surv_time = None, group_col = None, group_val = None ):

        find_median = self._build_median_func(surv_df, surv_time)
        input_df['Med_Surv'] = input_df.index.map(find_median)
        max_surv_time = np.max(input_df[duration_col])
        number_patients = input_df.shape[0]
        event_indicator = input_df[event_col].values 
        event_time      = input_df[duration_col].values + np.random.uniform(-0.03, 0.03, size = number_patients)*(input_df[duration_col].values < max_surv_time)
        estimated_risk  = -input_df['Med_Surv'].values 
        group_indicator = (input_df[group_col] == group_val).values

        sort_mask = np.argsort(event_time)
        event_time = event_time[sort_mask]
        event_indicator = event_indicator[sort_mask]
        estimated_risk = estimated_risk[sort_mask]
        group_indicator = group_indicator[sort_mask]
        
        idx = 0        
        current_time = event_time[idx]
        total_pairs   = 0
        correct_pairs = 0
        tied_pairs    = 0
        while current_time < max_surv_time and idx < number_patients-1:
            while (group_indicator[idx] == False or event_indicator[idx] == False) and idx < number_patients-1:
                idx += 1
            if idx == number_patients-1:
                break
            current_risk = estimated_risk[idx]
            total_pairs += np.sum(group_indicator[idx+1:])
            correct_pairs += np.sum((estimated_risk[idx+1:] < current_risk) * group_indicator[idx+1:] )
            tied_pairs += np.sum((estimated_risk[idx+1:] == current_risk)  * group_indicator[idx+1:] )

            idx +=1
        #print("Total pairs {}, correct_pairs {}, tied_pairs {}".format(total_pairs,correct_pairs,tied_pairs))
        return (correct_pairs + 0.5*tied_pairs)/total_pairs
    
    def _calculate_between_c_index(self, surv_df, input_df, duration_col='GTIME_KI', event_col='GSTATUS_KI', surv_time = None, group_col = None, group_val = None ):

        find_median = self._build_median_func(surv_df, surv_time)
        input_df['Med_Surv'] = input_df.index.map(find_median)
        max_surv_time = np.max(input_df[duration_col])
        number_patients = input_df.shape[0]
        event_indicator = input_df[event_col].values 
        event_time      = input_df[duration_col].values + np.random.uniform(-0.03, 0.03, size = number_patients)*(input_df[duration_col].values < max_surv_time)
        # estimated_risk  = -input_df['Med_Surv'].values + np.random.uniform(-0.03, 0.03, size = number_patients)*(input_df['Med_Surv'].values < max_surv_time)
        estimated_risk  = -input_df['Med_Surv'].values 
        group_indicator = (input_df[group_col] == group_val).values

        sort_mask = np.argsort(event_time)
        event_time = event_time[sort_mask]
        event_indicator = event_indicator[sort_mask]
        estimated_risk = estimated_risk[sort_mask]
        group_indicator = group_indicator[sort_mask]
        
        idx = 0        
        current_time = event_time[idx]
        total_pairs   = 0
        correct_pairs = 0
        tied_pairs    = 0
        while current_time < max_surv_time and idx < number_patients-1:
            while (event_indicator[idx] == False) and idx < number_patients-1:
                idx += 1
            if idx == number_patients-1:
                break
            current_risk = estimated_risk[idx]
            current_is_group = group_indicator[idx]
            if current_is_group:
                total_pairs += np.sum(~group_indicator[idx+1:])
                correct_pairs += np.sum((estimated_risk[idx+1:] < current_risk)*(~group_indicator[idx+1:]) )
                tied_pairs += np.sum((estimated_risk[idx+1:] == current_risk)*(~group_indicator[idx+1:]) )
            else:
                total_pairs += np.sum(group_indicator[idx+1:])
                correct_pairs += np.sum((estimated_risk[idx+1:] < current_risk) * group_indicator[idx+1:] )
                tied_pairs += np.sum((estimated_risk[idx+1:] == current_risk) *  group_indicator[idx+1:])
            idx +=1
        #print("Total pairs {}, correct_pairs {}, tied_pairs {}".format(total_pairs,correct_pairs,tied_pairs))
        if total_pairs==0: 
            return np.nan
        else:
            return (correct_pairs + 0.5*tied_pairs)/total_pairs
            
    def _calculate_brier_score(self, surv_df, input_df, days = int(365.25*5), duration_col='GTIME_KI', event_col='GSTATUS_KI', train_df = None):
        """Calculates brier score at time = days"""

        if train_df is None:
            train_df = input_df.copy()

        input_df = input_df.copy()
        
        # find the closest day in the survival dataframe to use 
        # we take the pessimist view and always take a value that is at least as big as the input day.
        
        d = 0 
        for day in surv_df.index:
            d = day
            if day >= days:
                break


        def get_surv(pt):
            """helper function to map a patient to a survival probability for a given survival time d"""
            pt =str(pt)
            if pt not in surv_df.columns:
                return np.nan
            return surv_df.loc[d, pt]

        # we add the survival probability to the input date       
        input_df['Surv'] = input_df.index.map(get_surv)
        
        
        #ToDo: What happens if the day input is smaller than the time to event for a patient? Should we change its value to censored?
        y = input_df[[event_col, duration_col] ].to_records(index=False)
        y_train = train_df[[event_col, duration_col] ].to_records(index=False)
        predictions = input_df['Surv'].values
        # Calculate the brier score
                
        try:
            score = brier_score(y_train, y, predictions, days)[1][0]
            return score
        except Exception as error: 
            print(error)
            return np.nan
        
    
    def _calculate_auc(self,surv_df, input_df, days = 365.25*5, duration_col='GTIME_KI', event_col='GSTATUS_KI', train_df = None):
        """Calculates AUC score at time = days"""

        if train_df is None:
            train_df = input_df
         
        #surv_df, input_df = self._read_output_file()
        #surv_df, input_df = self._fake_output_data()
        
        # find the closest day in the survival dataframe to use 
        # we take the pessimist view and always take a value that is at least as big as the input day.
        #input_df = input_df.copy(deep=True)
        input_df = copy.deepcopy(input_df)
         
        print(input_df.shape)
        input_df = input_df[input_df[duration_col]<input_df[duration_col].max()]
        print(input_df.shape)
        find_median = self._build_median_func(surv_df, days)
        # we add the survival probability to the input date
        input_df['Surv'] = input_df.index.map(find_median)
        #ToDo: What happens if the day input is smaller than the time to event for a patient? Should we change its value to censored?
       
        y = input_df[[event_col, duration_col]].to_records(index=False)
        
        print(input_df[[event_col, duration_col]])
        y_train = train_df[[event_col, duration_col] ].to_records(index=False)
        
        
        predictions = -input_df['Surv'].values
        # Calculate auc
        try:
            score = cumulative_dynamic_auc(y_train, y, predictions, days)[0][0]
            return score
        except Exception as error:
            #raise ValueError(error)
            return np.nan 
            
    def _calculate_mean_error(self,surv_df, input_df, days = 365.25*5, duration_col='GTIME_KI', event_col='GSTATUS_KI', train_df = None):
        """Calculates mean error at time = days"""

        input_df['censoring_weight'] = 1
        if train_df is not None:
            y_train = train_df[[event_col, duration_col] ].to_records(index=False)
            cens = CensoringDistributionEstimator().fit(y_train)
            prob_cens_t = cens.predict_proba([days])[0]
            prob_cens_y = cens.predict_proba(input_df[duration_col])
            prob_cens_y[prob_cens_y == 0] = np.inf
            input_df['inverse_cens'] = prob_cens_y
            input_df['censoring_weight'] = input_df.apply(lambda x: 1/x['censoring_weight'] if (x[event_col] and x[duration_col] <= days) else 1/prob_cens_t , axis=1)
            
         
        #surv_df, input_df = self._read_output_file()
        #surv_df, input_df = self._fake_output_data()
        
        # find the closest day in the survival dataframe to use 
        # we take the pessimist view and always take a value that is at least as big as the input day.
        input_df = input_df.copy()
        find_median = self._build_median_func(surv_df, days)
        # we add the survival probability to the input date
        input_df['Surv_Prob'] = input_df.index.map(find_median)
        input_df['t'] = days
        input_df['flag'] = input_df.apply(lambda x: np.nan if (not x[event_col]) and x[duration_col]<x['t'] else 1 , axis=1)
        
        input_df['surv_flag'] = input_df.apply(lambda x: (x[duration_col] >= x['t'])*x['flag'], axis=1 )
        return input_df.apply(lambda x: x['censoring_weight']*(x['surv_flag'] - x['Surv_Prob']), axis=1).mean()
    
    def _calculate_mean_squared_error(self,surv_df, input_df, days = 365.25*5, duration_col='GTIME_KI', event_col='GSTATUS_KI'):
        """Calculates mean error at time = days"""
         
        #surv_df, input_df = self._read_output_file()
        #surv_df, input_df = self._fake_output_data()
        
        # find the closest day in the survival dataframe to use 
        # we take the pessimist view and always take a value that is at least as big as the input day.
        input_df = input_df.copy()
        find_median = self._build_median_func(surv_df, days)
        # we add the survival probability to the input date
        input_df['Surv_Prob'] = input_df.index.map(find_median)
        input_df['t'] = days
        input_df['flag'] = input_df.apply(lambda x: np.nan if (not x[event_col]) and x[duration_col]<x['t'] else 1 , axis=1)
        input_df['surv_flag'] = input_df.apply(lambda x: (x[duration_col] >= x['t'])*x['flag'], axis=1 )
        return input_df.apply(lambda x: np.square(x['surv_flag'] - x['Surv_Prob']), axis=1).mean()

    def _create_test_subgroup_dataframe(self):
        file = os.listdir('data/'+self.path_name)[0]
        input_df = pd.read_csv('data/'+self.path_name+'/'+file)
        return input_df['DIAB'].apply(str)

    def _read_output_file(self):
        """returns DataFrames with input data used to calculate survival times and the predicted survival times"""
        raise NotImplementedError
        return None, None
    
    def _fake_output_data(self):
        """Builds a fake dataset for testing purposes
           returns surv_df, input_df"""
        days = np.arange(18) + 1
        ids  = [str(x) for x in np.arange(16) + 1]
        T = [1,2,3,4,6,7,7,8,10,14,16,18,18,18,18,18]
        E = [True,True,False,True,False,True,True,False,True,False,True,True,False,False,False,False]
        G = [True,False,False,True,True,True,False,True,False,False,False,True,False,True,True,False]
        Surv = [2,7,3,6,7,6,8,9,8,12,12,18,17,19,15,21]
        return pd.DataFrame({
            'Days': days,
            **{idx: [Surv[int(idx)-1] for _ in days] for idx in ids}
        }).set_index('Days'), \
            pd.DataFrame({
            'PT_CODE': ids,
            'T' : T,
            'E' : E, 
            'G' : G
        }).set_index('PT_CODE')

