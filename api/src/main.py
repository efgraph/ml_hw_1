import pickle
from pathlib import Path
import pandas as pd
from fastapi.responses import ORJSONResponse
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent

app = FastAPI(
    docs_url='/api/openapi',
    openapi_url='/api/openapi.json',
    default_response_class=ORJSONResponse,
)
model = pickle.load(open(f"{project_root}/models/model.pkl", "rb"))
train_encoder = pickle.load(open(f"{project_root}/models/train_encoder.pkl", "rb"))
train_scaler = pickle.load(open(f"{project_root}/models/train_scaler.pkl", "rb"))
train_loo_encoder = pickle.load(open(f"{project_root}/models/train_loo_encoder.pkl", "rb"))


def add_new_features(frames):
    for frame in frames:
        frame['age'] = 2023 - frame['year'] + 1
        frame['km_per_year'] = frame['km_driven'] / frame['age']
        frame['max_power_per_liter'] = frame['max_power'] / frame['engine'] * 1000
        frame['fresh_owner'] = frame['owner'].apply(lambda x: 1 if x in ['First Owner', 'Second Owner', 'Test Drive Car'] else 0)


def units_convert(frames):
    for frame in frames:
        frame['mileage'] = frame.mileage.str.replace(r'kmpl|km/kg', '', regex=True).astype(float)
        frame['engine'] = frame.engine.str.replace('CC', '').astype(float)
        frame['max_power'] = frame.max_power.str.replace('bhp', '')
        frame['max_power'] = frame.max_power.astype(str).apply(
            lambda x: np.NAN if x.strip() == '' else float(x)).astype(float)


def converter_max_torque(x):
    a = re.findall(r'(\d*\.?\d+)', x)
    if not len(a):
        return np.NAN
    return a[1] + a[2] if '+/-' in x else a[-1]


def converter_torque(x):
    a = re.findall(r'(\d*\.?\d+)', x)
    if not len(a):
        return np.NAN
    return round(float(a[0]) * 9.80665, 2) if 'kg' in x else a[0]


def convert_torque(frames):
    for frame in frames:
        frame['torque'] = frame['torque'].astype(str).apply(lambda x: x.replace(',', ''))
        frame['max_torque_rpm'] = frame['torque'].apply(converter_max_torque).astype(float)
        frame['torque'] = frame['torque'].apply(converter_torque).astype(float)


def convert(frames, columns, to_type):
    for df in frames:
        for column in columns:
            df[column] = df[column].astype(to_type)


def get_category_feat_seats(frame: pd.DataFrame):
    return frame.select_dtypes(include=['object']).join(frame['seats'])


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


def process_df(df: pd.DataFrame):
    units_convert([df])
    convert_torque([df])
    convert([df], ['engine', 'seats'], int)
    df_name = df['name']
    df = df.drop(['name', 'selling_price'], axis=1)
    add_new_features([df])
    df_category = get_category_feat_seats(df)
    df_category_encoded = train_encoder.transform(df_category)
    encoded_columns = train_encoder.get_feature_names_out(df_category.columns)
    df_encoded = pd.DataFrame.sparse.from_spmatrix(df_category_encoded, columns=encoded_columns)
    df_numeric = df[df.columns.difference(df_category.columns)]
    df_numeric['name'] = train_loo_encoder.transform(df_name)
    df_numeric = pd.DataFrame(train_scaler.transform(df_numeric),
                              index=df_numeric.index, columns=df_numeric.columns)
    df_encoded = df_encoded.join(df_numeric)
    return df_encoded


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    dict_values = item.dict()
    df = pd.DataFrame(dict_values, index=[0])
    df_encoded = process_df(df)
    result = model.predict(df_encoded)
    result = np.round(np.exp(result))
    return result


@app.post("/predict_items")
def upload(file: UploadFile = File(...)) -> List[dict]:
    df = pd.read_csv(file.file)
    df = df.drop(df.columns.difference(Item.__fields__), axis=1)
    df_encoded = process_df(df)
    result = model.predict(df_encoded)
    result = np.round(np.exp(result))
    df_result = pd.DataFrame(result, columns=['predicted_price'])
    df = df.join(df_result)
    resp = df.to_dict('records')
    return resp