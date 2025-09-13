from dash import Dash, dcc, html, Input, Output,State
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder  
import numpy as np
import os
import joblib
from Regression import Normal

app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH])




# Get directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build path to CSV relative to current file
csv_path = os.path.join(BASE_DIR, "Model", "Cars.csv")

# Read CSV
vehicle_df = pd.read_csv(csv_path)


num_cols = ["year", "max_power", "mileage", "km_driven"]

# Build paths to model files
model_path = os.path.join(BASE_DIR, "Model_New", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "Model", "Model", "car-scalar.model")
brand_label_path = os.path.join(BASE_DIR, "Model", "Model", "brand-label.model")
fuel_label_path = os.path.join(BASE_DIR, "Model", "Model", "brand-fuel.model")

"""
# Replace pickle loading with joblib
try:
    model = joblib.load("code/Model_New/model.model")
except Exception as e:
    print(f"Error loading model with joblib: {e}")

"""


# Load models
with open(model_path, "rb") as f:
    model = pickle.load(f)


with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(brand_label_path, 'rb') as f:
    label_brand_model = pickle.load(f)

with open(fuel_label_path, 'rb') as f:
    label_fuel_model = pickle.load(f)

brand_cat = list(label_brand_model.classes_)
fuel_cat = list(label_fuel_model.classes_)


default_values = {
    'year': 2017,
    'max_power': 82.4,
    'brand': 'Maruti',
    'mileage': 19.42,
    'fuel': 'Diesel',
    'km_driven':100000

}
# -----------------------------
# Input Card
# -----------------------------
app.layout = dbc.Container(
    dbc.Row(
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H1(" Car Price Prediction", className="text-center mb-4"),
                    html.Hr(),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Brand"),
                            dcc.Dropdown(id="brand", options=brand_cat, value=brand_cat[0])
                        ], width=4),

                        dbc.Col([
                            dbc.Label("Year"),
                            dcc.Dropdown(
                                id="year",
                                options=[{"label": y, "value": y} for y in sorted(vehicle_df['year'].unique())],
                                value=vehicle_df['year'].min()
                            )
                        ], width=4),

                        dbc.Col([
                            dbc.Label("Fuel"),
                            dcc.Dropdown(id="fuel", options=fuel_cat, value=fuel_cat[0])
                        ], width=4),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Mileage (km/l)"),
                            dcc.Input(id="mileage", type="number", value=0, style={"width": "100%"})
                        ], width=6),

                        dbc.Col([
                            dbc.Label("km_driven"),
                            dcc.Input(id="km_driven", type="number", value=0, style={"width": "100%"})
                        ], width=6),

                        dbc.Col([
                            dbc.Label("Max Power (bhp)"),
                            dcc.Input(id="max_power", type="number", value=0, style={"width": "100%"})
                        ], width=6),
                    ], className="mb-3"),

                        

                    dbc.Button("Predict Price", id="submit", color="primary", className="w-100 mb-3"),
                    html.Div(id="prediction_result", className="text-center fs-4 fw-bold")
                ]),
                className="shadow p-4"
            ),
            width=8,  # make card 8/12 of the page width
            className="mx-auto my-5"  # center horizontally and add vertical margin
        )
    ),
    fluid=True
)


# -----------------------------
# Output Card
# -----------------------------
output_card = dbc.Card(
    dbc.CardBody([
        html.H4("Car Price Prediction", className="card-title", style={'textAlign':'center'}),
        html.Div(id='prediction-output', style={'fontSize': 28, 'color': '#2B8CFF', 'textAlign': 'center', 'fontWeight':'bold'})
    ]),
    className="mt-4 shadow-sm"
)


# Callback
# -----------------------------
@app.callback(
    Output("prediction_result", "children"),
    Input("submit", "n_clicks"),
    State("year", "value"),
    State("max_power", "value"),
    State("brand", "value"),
    State("mileage", "value"),
    State("fuel", "value"), 
    State("km_driven", "value"),
    prevent_initial_call=True
)
def predict_price(n, year, max_power, brand, mileage, fuel, km_driven):
    try:
        # Step 1: Collect features
        features = {            
            "max_power": max_power,
            "mileage": mileage,
            "fuel": fuel,
            "brand": brand,            
            "year": year,
            "km_driven": km_driven,
        }

        # Step 2: Fill missing/invalid
        for f in features:
            if features[f] is None or features[f] == "":
                features[f] = default_values[f]
            elif f in num_cols and features[f] < 0:
                features[f] = default_values[f]

        # Step 3: DataFrame
        X = pd.DataFrame([features])
        # Step 1: Encode categorical first
        X["brand"] = label_brand_model.transform([X["brand"].iloc[0]])
        X["fuel"] = label_fuel_model.transform([X["fuel"].iloc[0]])


        


        scaler_cols = ['max_power','mileage','fuel','brand','year','km_driven']  # must match training
        X[scaler_cols] = scaler.transform(X[scaler_cols])

 

        # Step 6: Predict
        y_pred = model.predict(X)
        try:
            price = np.round(np.exp(y_pred), 2)[0]  # if model trained on log(price)
        except OverflowError:
            price = np.round(y_pred, 2)[0]

        return f"💰 Predicted Price: ${price}"

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



