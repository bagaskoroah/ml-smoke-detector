import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from utils import *

from pathlib import Path
project_root = Path().resolve().parent

# load config
config = load_config()

# define data validation function
def data_validation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Do data validation for removing bad data.

    Parameters:
    ----------
    data : pd.DataFrame
        Loaded raw dataset.

    Returns:
    -------
    data : pd.DataFrame
        Validated data.
    """    
    # convert utc column into datetime type
    data[config['column_datetime']] = pd.to_datetime(data[config['column_datetime']], unit='s')
    
    # drop the CNT column
    data = data.drop(columns=['CNT'])

    # rename the column
    new_column = ['utc', 'temperature', 'humidity_pct', 'tvoc', 'co2', 'raw_h2', 'raw_ethanol', 'pressure', 'pm10', 'pm25', 'nc05', 'nc10', 'nc25', 'fire_alarm']
    data.columns = new_column

    return data

# define data defense function
def data_defense(data: pd.DataFrame, config: dict, api=False):
    """
    Do data defense to check data types and range.

    Parameters:
    ----------
    data : pd.DataFrame
        Validated data.

    config : dict
        Loaded configuration file.

    api : bool, default = False
        To check whether the input data from API or not.

    Returns:
    -------
    None, its a void function.
    """
    # ensure raw data and config is immutable
    data = data.copy()
    config = copy.deepcopy(config)

    # number of data
    n_data = len(data)

    # list of columns
    cols_float = config['columns_float']
    cols_int = config['columns_int']

    # if the input is not from API
    if not api:
        # check data types
        assert data.select_dtypes('float').columns.tolist() == cols_float, 'an error occurs in float columns.'
        assert data.select_dtypes('int').columns.tolist() == cols_int, 'an error occurs in int columns.'
        
        # check range values of data
        for col in config['features']:
            min_value = config[f'range_{col}'][0]
            max_value = config[f'range_{col}'][1]
            assert data[col].between(min_value, max_value).sum() == n_data, f'an error occurs in {col} range'

    else:
        # Float features used only temperature, humidity_pct, pressure, and pm10. 
        del cols_float[4:]

        # Int features used only tvoc, co2, raw_h2, and raw_ethanol.
        del cols_int[4:]
        
        api_data = data[cols_float+cols_int]

        # check data types
        assert api_data.select_dtypes('float').columns.tolist() == cols_float, 'an error occurs in float columns.'
        assert api_data.select_dtypes('int').columns.tolist() == cols_int, 'an error occurs in int columns.'

        # check range of data
        for col in (cols_float+cols_int):
            min_value = config[f'range_{col}'][0]
            max_value = config[f'range_{col}'][1]
            assert data[col].between(min_value, max_value).sum() == n_data, f'an error occurs in {col} range'            

# define an input-output split function
def split_input_output(data, config):
    """
    Split the input(X) and output (y).

    Parameters:
    ----------
    data : pd.DataFrame
        The processed dataset.

    config : dict
        Loaded configuration parameters.

    Returns:
    -------
    X : pd.DataFrame
        The input data.

    y : pd.Series
        The output data.    
    """

    # ensure raw data immutable
    data = data.copy()

    # split x and y
    X = data[config['features']]
    y = data[config['label']]

    return X, y

# define a train-test split function
def split_train_test(X, y, test_size, random_state=config['random_state']):
    """
    Split the train and test set.

    Parameters:
    ----------
    X : pd.DataFrame
        The input data.

    y : pd.Series
        The output data.

    test_size : float
        The proportion of test set.

    random_state : int, default = 123
        For reproducibility

    Returns:
    -------
    X_train, X_test : pd.DataFrame
        The train and test input.

    y_train, y_test : pd.Series
        The train and test output.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

# main function
def main():
    # load raw dataset
    raw_data = load_data(path_data=project_root/config['raw_data'])
    print('Raw data is loaded. . .')

    # data validation
    validated_data = data_validation(data=raw_data)
    print('Data has successfully validated. . .')

    # serialize validated data
    serialize_data(data=validated_data, path=project_root/'interim/validated_data.pkl')
    print('Validated data is serialized. . .')

    # data defense
    data_defense(data=validated_data, config=config)
    print('Data defense mechanism has successfully done. . .')

    # split input-output
    X, y = split_input_output(data=validated_data, config=config)

    # split train, valid, and test
    X_train, X_not_train, y_train, y_not_train = split_train_test(
        X=X,
        y=y,
        test_size=config['size_train_valid'],
        random_state=config['random_state']
    )

    X_valid, X_test, y_valid, y_test = split_train_test(
        X=X_not_train,
        y=y_not_train,
        test_size=config['size_valid_test'],
        random_state=config['random_state']
    )

    print('Split train-valid-test has been successfully done. . .')

    # data serialization
    serialize_data(X_train, project_root/config['path_data_X_train'])
    serialize_data(y_train, project_root/config['path_data_y_train'])
    serialize_data(X_valid, project_root/config['path_data_X_valid'])
    serialize_data(y_valid, project_root/config['path_data_y_valid'])
    serialize_data(X_test, project_root/config['path_data_X_test'])
    serialize_data(y_test, project_root/config['path_data_y_test'])
    print('Data pipeline has been successfully saved. . .')

if __name__ == '__main__':
    main()