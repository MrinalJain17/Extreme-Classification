import pandas as pd
import re

def dict_spliter(line):
    temp_list = re.split(r'[:\s]\s*', line)
    return {int(temp_list[i]): float(temp_list[i + 1]) for i in range(0, len(temp_list)-1, 2)} 

def load_data(file_name):
    df = pd.read_csv(file_name, index_col  = 'ex_id')
    df.labels = df.labels.str.split(',')
    df.features = df.features.apply(dict_spliter)
    return df