# import needed libraries
import yaml
import joblib
from pathlib import Path
import pandas as pd

# define project root path
project_root = Path().resolve().parent

# define config path
config_path = project_root/'config/config.yaml'

def load_config() -> dict:
    """ Load the configuration file (config.yaml) """

    # check if there is any config file in path
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    # raise error if there is no config file in path
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file can't be found in the {config_path}.")

    return config

def update_config(key, value, config):
    """
    Update the configuration parameter values.

    Parameters:
    ----------
    key : str
        Key to be updated.

    value : any type supported in Python
        Updated value.

    config : dict
        Loaded configuration file.

    Returns:
    -------
    config : dict
        Updated configuration file.
    """
    # ensure raw config file remain immutable
    config = config.copy()

    # update configuration parameters
    config[key] = value

    # rewrite config file
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    # notify that config has been updated
    print(f'Config has been successfully updated. \nKey: {key} \nValue: {value}\n')

    # reload and return config
    config = load_config()
    return config

def load_data(fname: str) -> pd.DataFrame:
    '''
    Load dataframe and its data shape information.

    Param:
    fname <str> : raw dataset path to be loaded.

    Return:
    <pd.DataFrame> : loaded dataframe.
    '''
    data = pd.read_csv(fname)
    print('Data Shape:', data.shape)
    return data

def serialize_data(data, path):
    """
    Dump data into pickle file.

    Parameters:
    data : pd.DataFrame or sklearn object
        Data to be serialize.

    path : str
        Serialized data location.

    Returns:
    -------
    None, its a void function.
    """

    print(f"Data serialized to {path}")
    return joblib.dump(data, path)

def deserialize_data(path):
    """
    Load and return pickle file.

    Parameters:
    ----------
    path : str
        Serialized data location.

    Returns:
    -------
    pd.DataFrame or sklearn object.
    """

    print(f"Data deserialized from {path}")
    return joblib.load(path)