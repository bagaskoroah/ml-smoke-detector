from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn
import pandas as pd

import utils
import data_pipeline
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

# serialize needed components
config = utils.load_config()
best_model = utils.deserialize_data(path=project_root/config['path_best_model'])

# define input data structure
class DataAPI(BaseModel):
    """Represents the user input data structure."""
    temperature: float
    humidity_pct: float
    pressure: float
    pm10: float
    tvoc: int
    co2: int
    raw_h2: int
    raw_ethanol: int

# create API object
app = FastAPI()

# define handlers
@app.get('/')
def home():
    return {'message': 'Hello, FastAPI up!'}

@app.post('/predict/')
def predict(data: DataAPI):
    # convert DataAPI to pandas dataframe.
    data = pd.DataFrame([data.dict()])

    # perform data defense
    try:
        data_pipeline.data_defense(data=data, config=config, api=True)
    except AssertionError as err:
        return {'res': [], 'error_msg': str(err)}
    
    # predict data
    y_pred = best_model.predict(data)

    if y_pred[0] == 0:
        y_pred = 'FIRE NOT DETECTED.'
    else:
        y_pred = 'WARNING: FIRE DETECTED!'
    
    return {'res': y_pred, 'error_msg': ''}

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8080)