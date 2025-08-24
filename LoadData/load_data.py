import csv
import numpy as np
import pandas as pd
from LoadData.verify_data import loaded_data_verification

FILE_NAME = './LoadData/spambase/spambase.data'

#########
## csv ##
#########
def load_data_csv(file_dir):
    with open(file_dir, 'r') as f:
        data = list(csv.reader(f, delimiter=','))
    return np.array(data)


data_from_csv = load_data_csv(FILE_NAME)
loaded_data_verification(data_from_csv)


###########
## numpy ##
###########
def load_data_np(file_dir, method='genfromtxt'):
    """
    - dtype
    - header
    - missing data
    """
    if method == 'loadtxt':
        return np.loadtxt(
            file_dir,
            delimiter=',',
            dtype=np.float32,
            skiprows=0,
            missing_values=''
        )
    elif method == 'genfromtxt':
        return np.genfromtxt(
            file_dir,
            delimiter=',',
            dtype=np.float32,
            skiprows=0,
            missing_values='',
            filling_values=999.0
        )


data_loadtxt = load_data_np(FILE_NAME, method='loadtxt')
loaded_data_verification(data_loadtxt)

data_genfromtxt = load_data_np(FILE_NAME, method='genfromtxt')
loaded_data_verification(data_genfromtxt)


############
## pandas ##
############
def load_data_pd(file_dir):
    """
    dtype
    header
    missing data
    """
    df = pd.read_csv(
        file_dir,
        delimiter=',',
        skiprows=0,
        na_values=['hello', 'world']
    )
    df.fillna(999.0)

    data = df.to_numpy()
    data = np.asarray(data, dtype=np.float32)

    return data

data_pd = load_data_pd(FILE_NAME)
loaded_data_verification(data_pd)