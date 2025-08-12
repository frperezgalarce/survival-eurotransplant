import numpy as np
from src.t_star import TStar 
from src.utils import *
from pathlib import Path
import yaml 
from typing import List

class SubpopulationCreator:

    def __init__(self, t_star: TStar):
        
        if t_star.training_data is None or t_star.testing_data is None:
            raise("t_star has no data!")
        
        self.t_star = t_star
        with open(Path(__file__).parent.parent/'paths.yaml', 'r') as file:
            paths = yaml.safe_load(file)
        
        self.ROOT_PATH  = paths['PATH_CODE_DICT_ROOT']
        self.LOCAL_PATH = paths['PATH_CODE_DICT_LOCAL']
        self.file_name  = 'KIDPAN_FORMATS_FLATFILE'

        self.dict_df = get_data(self.ROOT_PATH, self.LOCAL_PATH, self.file_name)

    def create_eth(self):
        col_name= 'ETHCAT'
        code_dict = self.dict_df[self.dict_df['FMTNAME'] == col_name].set_index('Code').to_dict()['label']

        def mapping(eth_code):
            eth_code = str(eth_code)
            if eth_code in [np.nan, '.', '..']:
                return 'Not Reported'
            if eth_code not in code_dict.keys():
                return 'Not Reported'
            return code_dict[eth_code]
        
        training = self.t_star.training_data.copy()
        testing  = self.t_star.testing_data.copy()
        training['ETHNICITY'] = training[col_name].apply(mapping)
        testing['ETHNICITY']  = testing[col_name].apply(mapping)
        return training['ETHNICITY'], testing['ETHNICITY']
    
    def create_gender(self):
        training = self.t_star.training_data.copy()
        testing  = self.t_star.testing_data.copy()

        return training['GENDER'], testing['GENDER']
    
    def create_ages(self, breaks: List[int]):
        """
            breaks: list of ints to create groups. Each interval is right opened: [b, b+1)
        """
        training = self.t_star.training_data.copy()
        testing  = self.t_star.testing_data.copy()

        intervals = [[a,b] for (a,b) in zip([0] + breaks, breaks + [999])]
        labels = ['{}-{} years'.format(x[0],x[1]).replace('999', '') for x in intervals]
        labels[-1] = '>= {} years'.format(breaks[-1])

        def age_to_group(age):
            for (i,x) in enumerate(breaks):
                if age < x:
                    return labels[i]
            return labels[-1]

        training['AGE_GROUP'] = training['AGE'].astype(int).apply(age_to_group)
        testing['AGE_GROUP']  = testing['AGE'].astype(int).apply(age_to_group)
        return training['AGE_GROUP'], testing['AGE_GROUP']
    
    def create_blood_type(self):
        """
        TODO: Should some groups be combined?
        """
        def simplify_abo(abo):
            if 'A' in abo and 'B' in abo:
                return 'AB'
            elif 'A' in abo:
                return 'A'
            elif 'B' in abo:
                return 'B'
            else:
                return 'O' 
                
        training = self.t_star.training_data.copy()
        testing  = self.t_star.testing_data.copy()

        training['ABO'] = training['ABO'].apply(simplify_abo)
        testing['ABO'] = testing['ABO'].apply(simplify_abo)
        
        return training['ABO'], testing['ABO']

