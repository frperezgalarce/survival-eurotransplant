import os 
import pandas as pd
from scipy.stats import ttest_ind, f_oneway, normaltest
from scipy.stats import mannwhitneyu


def tests(df, on = 'pv', by='Concordance_index', set='test', normality=False):
    results = []
    for a in df[on].unique():
        for b in df[on].unique():
            # Get Brier scores for each pair
            set_a = df[(df['set'] == set) & (df[on] == a)][by].values

            
            set_b = df[(df['set'] == set) & (df[on] == b)][by].values

            
            if float(a) != float(b) and float(a)<float(b):
                
                
                res = normaltest(set_a)
                print(a)             
                # null hypothesis that a sample comes from a normal distribution   
                print('normal test set a- pvalue: ', res.pvalue)
                
                
                res = normaltest(set_b)
                print(b)
                print('normal test set b - pvalue: ', res.pvalue)
                # Perform independent t-test
                if normality:
                    stat, pvalue = ttest_ind(set_a, set_b, equal_var=False)
                else: 
                    stat, pvalue = mannwhitneyu(set_a, set_b, method="exact")
                    
                # Perform F-test for equality of variances
                f_stat = set_a.var(ddof=1) / set_b.var(ddof=1)
                f_pvalue = f_oneway(set_a, set_b).pvalue

                # Determine if null hypothesis is rejected
                t_reject_null = pvalue < 0.05
                f_reject_null = f_pvalue < 0.05

                # Append results with rejection status
                results.append({
                    on+'_a': a,
                    on+'_b': b,
                    'stat': stat,
                    'pvalue': pvalue,
                    't_reject_null': t_reject_null,
                    'f_stat': f_stat,
                    'f_pvalue': f_pvalue,
                    'f_reject_null': f_reject_null,
                    'var_a': set_a.var(ddof=1),
                    'var_b': set_b.var(ddof=1),
                    'mean_a': set_a.mean(),
                    'mean_b': set_b.mean()
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def get_complete_table(group, metric, t_star_year, txtp, ept, ppath): 
    TRAIN_RESULTS_PATH = os.path.join(ppath, 'notebooks/experiments/results/train')
    TEST_RESULTS_PATH = os.path.join(ppath, 'notebooks/experiments/results/test')

    files_test = get_files(TEST_RESULTS_PATH, group, metric, t_star_year, txtp, ept)
    files_train = get_files(TRAIN_RESULTS_PATH, group, metric, t_star_year, txtp, ept)

    if not files_train and not files_test:
        print(f"No train/test files found for: {group}, {metric}, {t_star_year}, {txtp}, {ept}")
        return pd.DataFrame()

    df_train = append_results(TRAIN_RESULTS_PATH, files_train) if files_train else pd.DataFrame()
    df_test = append_results(TEST_RESULTS_PATH, files_test) if files_test else pd.DataFrame()

    df_complete_train = unroll_columns(df_train) if not df_train.empty else pd.DataFrame()
    df_complete_test = unroll_columns(df_test) if not df_test.empty else pd.DataFrame()

    df_merged = pd.concat([df_complete_train, df_complete_test], ignore_index=True)
    
    return df_merged

def get_files(path, group = 'ETHNICITY', metric = 'brier', 
              t_star = '2004', txpt = 'txpt_1', 
              ept = 'ept_5'):
    prediction_test_files = [f for f in os.listdir(path) if 
                                (group in f) 
                            and (metric in f)
                            and (t_star in f)
                            and (txpt in f)
                            and (ept in f)]   
    return prediction_test_files

def append_results(path, files):
    df = pd.DataFrame() 
    for f in files: 
        new_df = pd.read_csv(path+'/'+f)
        new_df['file'] = f
        df = pd.concat([df, new_df])
    return df 

def unroll_columns(df): 
    split_data = df['file'].str.split('_', expand=True)
    columns = {}
    for i in range(0, split_data.shape[1], 2):
        col_name = split_data[i]       # Even-indexed column: column name
        if i+1 < split_data.shape[1]:
            col_value = split_data[i+1]  # Odd-indexed column: value
        else:
            col_value = pd.Series([None]*len(df))
        columns[col_name.iloc[0]] = col_value  # Use the first entry to name the column. 
    new_cols_df = pd.DataFrame(columns)
    df = pd.concat([df, new_cols_df], axis=1)
    df['pop'] = df['pop'].str.replace('.csv','')
    del df['file']
    return df