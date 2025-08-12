from typing import Optional
import pandas as pd
import os
from tqdm import tqdm
import time 
import numpy as np 
from scipy import stats
from src.metrics import Metrics

          
def get_data(ROOT_PATH: str, LOCAL_PATH: str, file_name: str) -> Optional[pd.DataFrame]:
    file_path = ROOT_PATH + LOCAL_PATH + file_name + '.htm'  # Adjust the path as needed
    
    tables = pd.read_html(file_path)
    html_table = tables[0] if tables else None

    if html_table is not None:
        df_columns = html_table['LABEL'].to_list()  # Assuming 'LABEL' column contains the column names
        
        df = pd.read_table(ROOT_PATH + LOCAL_PATH + file_name + '.DAT',
                          header=None, low_memory=False, index_col=None)

        df.columns = df_columns  

        return df
    else:
        return None
    
def load_kidpan_donor(mode='load', save=False, ROOT_PATH=None, 
                      date ='12/31/2015', filter_death=False, 
                      DON_TYP='C', final_date=None, 
                      competing_risks=False, build_dataset=True):
    '''
    DON_TYP: type of donor, L is living donor and C is deceased donor.
    '''

    if DON_TYP not in ['L', 'C']: 
        raise ValueError('Type of donor not included.')

    if mode=='load':
        df_kidpan = pd.read_csv('data/kidpan.csv')
        df_donor = pd.read_csv('data/'+DON_TYP+'_donor.csv')
        
    elif mode=='create':
        LOCAL_PATH = 'Kidney_ Pancreas_ Kidney-Pancreas/'
        file_name = 'KIDPAN_DATA' 

        if build_dataset:
            print('building data set...')
            
            df_kidpan_raw = get_data(ROOT_PATH, LOCAL_PATH, file_name)
            
            columns = ['GTIME_KI','GSTATUS_KI', 'PT_CODE', 'DONOR_ID',  'TRR_ID_CODE', 
                        'TX_DATE', 
                       'AGE_DON', 'HCV_SEROSTATUS', 'GENDER_DON',  'AGE', 'DIAB', 'DIALYSIS_DATE',
                       'NUM_PREV_TX', 'BMI_CALC', 'ETHCAT', 'GENDER', 'ABO_MAT',
                       'AMIS', 'BMIS', 'NPKID',
                       'DRMIS','HLAMIS', 'COLD_ISCH_KI',  'ABO_DON', 'SERUM_CREAT', 'CREAT_TRR',
                       'COMPOSITE_DEATH_DATE','ACUTE_REJ_EPI_KI', 'ABO', 'DIABETES_DON', 'GRF_FAIL_CAUSE_TY_KI', 
                       'TOT_SERUM_ALBUM', 'DAYSWAIT_CHRON', 'DAYSWAIT_CHRON_KI', 
                       'PRI_PAYMENT_TCR_KI', 'DIAG_KI', 'DGN_TCR', 'PRE_TX_TXFUS', 'END_CPRA_DETAIL', 
                       'PERIP_VASC',  'BMI_DON_CALC', 'HIST_CIG_DON', 'CMV_IGG', 'TXKID', 'DISTANCE', 
                       'ON_DIALYSIS', 'DIAL_TRR', 'HGT_CM_CALC', 'WGT_KG_CALC', 'WL_ID_CODE', 
                       'RDA1', 'RDA2', 'RDB1', 'RDB2', 'RDDR1', 'RDDR2', 'DIABDUR_DON']



            df_kidpan_filtered = df_kidpan_raw[(df_kidpan_raw.ORGAN=='KI') & 
                                (df_kidpan_raw.AGE_GROUP=='A') & 
                                (df_kidpan_raw.DON_TY.isin([DON_TYP]))][columns] #D
            


            df_kidpan_filtered['NPKID'] = pd.to_numeric(df_kidpan_filtered['NPKID'], 
                                                        errors='coerce')
            df_kidpan_filtered['NUM_PREV_TX'] = pd.to_numeric(df_kidpan_filtered['NUM_PREV_TX'], 
                                                              errors='coerce')

            print(df_kidpan_filtered.shape)
            
            df_kidpan_filtered.NUM_PREV_TX = df_kidpan_filtered.NUM_PREV_TX.fillna(value=-1)
            df_kidpan_filtered.NPKID = df_kidpan_filtered.NPKID.fillna(value=-1)

            df_kidpan_filtered = df_kidpan_filtered[df_kidpan_filtered['NUM_PREV_TX'] == df_kidpan_filtered['NPKID']]
            
            df_kidpan_filtered.NUM_PREV_TX = df_kidpan_filtered.NUM_PREV_TX.fillna(value=-1)
            df_kidpan_filtered.NPKID = df_kidpan_filtered.NPKID.fillna(value=-1)
            
            replace_map = {-1: np.nan}
            df_kidpan_filtered[['NUM_PREV_TX', 'NPKID']] = df_kidpan_filtered[['NUM_PREV_TX', 'NPKID']].replace(replace_map)
            
            print(df_kidpan_filtered.shape)

            df_kidpan = delete_missing_values_protected_features(df_kidpan_filtered)
            
            df_kidpan_filtered.to_csv('data/kidpan_filtered.csv')
        
            
        else: 
            df_kidpan_filtered = pd.read_csv('data/kidpan_filtered.csv')
            df_kidpan = delete_missing_values_protected_features(df_kidpan_filtered)
                   
            
        df_kidpan['GSTATUS_KI'] = (df_kidpan['GSTATUS_KI'].astype(int).astype(bool))
        df_kidpan['GTIME_KI'] = df_kidpan['GTIME_KI'].astype(int)
        
        df_kidpan['TX_DATE'].replace('.', pd.NaT, inplace=True)
        
        df_kidpan['TX_DATE']= pd.to_datetime(df_kidpan['TX_DATE'], format='%m/%d/%Y', errors='coerce')
        
        df_kidpan['COMPOSITE_DEATH_DATE']= pd.to_datetime(df_kidpan['COMPOSITE_DEATH_DATE'],
                                                                 format='%m/%d/%Y', errors='coerce')
        
        df_kidpan = df_kidpan[df_kidpan['TX_DATE'] > date]
        
        if final_date: 
            df_kidpan = df_kidpan[df_kidpan['TX_DATE'] < final_date]


        if DON_TYP == 'L':
            LOCAL_PATH = 'Living Donor/'
            file_name = 'LIVING_DONOR_DATA' 
            df_donor = get_data(ROOT_PATH, LOCAL_PATH, file_name)
            df_donor = df_donor[['DONOR_ID', 'DIABETES','PREDON_HGT','PREDON_WGT', 'PREOP_CREAT_LI', 
                           'ETHCAT_DON', 'HIST_HYPER']]
            df_donor.columns = ['DONOR_ID', 'HIST_DIABETES_DON', 'HGT_CM_DON_CALC', 'WGT_KG_DON_CALC',
                            'CREAT_DON', 'ETHCAT_DON', 'HIST_HYPERTENS_DON']
        elif DON_TYP == 'C': 
            if build_dataset:
                LOCAL_PATH = 'Deceased Donor/'
                file_name = 'DECEASED_DONOR_DATA' 
                df_donor = get_data(ROOT_PATH, LOCAL_PATH, file_name)
                df_donor = df_donor[['DONOR_ID', 'HIST_DIABETES_DON', 'HGT_CM_DON_CALC', 'COD_CAD_DON', 
                                                'WGT_KG_DON_CALC', 'CREAT_DON', 'ETHCAT_DON',  'HIST_HYPERTENS_DON', 
                                                'NON_HRT_DON']]
                df_donor['CREAT_DON'] = pd.to_numeric(df_donor['CREAT_DON'], errors='coerce')

                df_donor.to_csv('data/deceased_donor.csv')
            else: 
                df_donor = pd.read_csv('data/deceased_donor.csv')
                
        if filter_death:
            df_kidpan =  df_kidpan[df_kidpan['COMPOSITE_DEATH_DATE'].isna()]

        if competing_risks:        
            df_kidpan['GSTATUS_KI'] = df_kidpan['GSTATUS_KI'] + (~df_kidpan['COMPOSITE_DEATH_DATE'].isna() & (df_kidpan['GSTATUS_KI']!=1))*2

        df_donor.DONOR_ID = df_donor.DONOR_ID.astype(int) 
        df_kidpan.DONOR_ID = df_kidpan.DONOR_ID.astype(int)
        
        if save: 
            df_kidpan.to_csv('data/kidpan.csv')
            df_donor.to_csv('data/'+DON_TYP+'_donor.csv')
    else: 
        raise ValueError('this mode does not exit, please chose load or create')

    return df_kidpan, df_donor

def delete_missing_values_protected_features(df): 
    df= df[df.GTIME_KI!='.'] 
    df= df[df.AGE!='.']
    df= df[df.ETHCAT!='.']
    df= df[df.ABO!='.']
    df= df[df.GENDER!='.']
    return df
        

def save_metrics(predictions, local_path, t_star_short, subpopulations, 
                 df, date_for_prediction, metrics, train_df = None): 
    
    df_inverse= df.copy()
         
    predictions.columns = [str(x) for x in predictions.columns]
    
    try:
        result_auc = metrics.auc(None, date_for_prediction, inputs={0: df}, predictions={0: [predictions]}, train_df = df_inverse)
        result_auc.to_csv(local_path+t_star_short+'_metric_auc_pop_global.csv')
    except Exception as error: 
        raise ValueError(error) 
    time.sleep(1)
    
    
    try:
        result = metrics.brier_score(None, date_for_prediction, inputs={0: df}, predictions={0: [predictions]},
                                 train_df = df_inverse)

        result.to_csv(local_path+t_star_short+'_metric_brier_pop_global.csv')

    except Exception as error:
        raise ValueError(error) 
        
    time.sleep(1)
    try:
        print(date_for_prediction)
        print(predictions.columns)
        print(df.index)
        result_cindex = metrics.concordance_index(subgroup_df = None, surv_time=date_for_prediction, inputs={0: df},
                                                predictions={0: [predictions]})
        print(result_cindex)
        result_cindex.to_csv(local_path+t_star_short+'_metric_cindex_pop_global.csv')
    except Exception as error: 
        raise ValueError(error) 
    
    time.sleep(1)    

    
    try:
        result_mse = metrics.mean_error(None, date_for_prediction, inputs={0: df}, predictions={0: [predictions]}, train_df = df_inverse)

        result_mse.to_csv(local_path+t_star_short+'_metric_mse_pop_global.csv')
    except Exception as error: 
        raise ValueError(error) 
    time.sleep(1)
    
    for sub in subpopulations:
        try:
            result = metrics.brier_score(sub, date_for_prediction, inputs={0: df}, predictions={0: [predictions]}, 
                                        visualize = False, train_df = df_inverse)
            
            result.to_csv(local_path+t_star_short+'_metric_brier_pop_'+sub.name+'.csv')
        except Exception as error: 
            raise ValueError(error) 
       
        time.sleep(1)
    
    
    for sub in subpopulations:
        try: 
            result = metrics.subgroup_c_index(sub, date_for_prediction, inputs={0: df}, predictions={0: [predictions]})
            result.to_csv(local_path+t_star_short+'_metric_cindex_pop_'+sub.name+'.csv')
        except Exception as error: 
            raise ValueError(error) 
        time.sleep(1)
            
    for sub in subpopulations:
        try:
            result = metrics.auc(sub, date_for_prediction, inputs={0: df}, predictions={0: [predictions]}, train_df = df_inverse)
            result.to_csv(local_path+t_star_short+'_metric_auc_pop_'+sub.name+'.csv')
        except Exception as error: 
            raise ValueError(error) 
        time.sleep(1)
            
    for sub in subpopulations:
        #sub.index = df.index
        try:
            result = metrics.mean_error(sub, date_for_prediction, inputs={0: df}, predictions={0: [predictions]}, train_df = df_inverse)
            result.to_csv(local_path+t_star_short+'_metric_mse_pop_'+sub.name+'.csv')
        except Exception as error:
            raise ValueError(error) 
        time.sleep(1)