# Import the required libraries.
import streamlit as st
import requests

import utils as utils

# load config
config = utils.load_config()

# Add some information about the service.
st.title("Smoke Detector Service")
st.subheader("Just enter the input below then click Predict button!")

# Create the input form.
with st.form(key = "data_form"):
    # Create input box for number input.
    min_temp, max_temp = float(config["range_temperature"][0]), float(config["range_temperature"][1])
    temp = st.number_input(
        label = "1.\tEnter temperature Value:",
        min_value = min_temp,
        max_value = max_temp,
        help = f"Value range from {min_temp} to {max_temp}"
    )

    min_humid, max_humid = float(config["range_humidity_pct"][0]), float(config["range_humidity_pct"][1])
    humid = st.number_input(
        label = "2.\tEnter humidity_pct Value:",
        min_value = min_humid,
        max_value = max_humid,
        help = f"Value range from {min_humid} to {max_humid}"
    )

    min_pressure, max_pressure = float(config["range_pressure"][0]), float(config["range_pressure"][1])
    pressure = st.number_input(
        label = "3.\tEnter pressure Value:",
        min_value = min_pressure,
        max_value = max_pressure,
        help = f"Value range from {min_pressure} to {max_pressure}"
    )

    min_pm10, max_pm10 = float(config["range_pm10"][0]), float(config["range_pm10"][1])
    pm10 = st.number_input(
        label = "4.\tEnter pm10 Value:",
        min_value = min_pm10,
        max_value = max_pm10,
        help = f"Value range from {min_pm10} to {max_pm10}"
    )

    min_tvoc, max_tvoc = int(config["range_tvoc"][0]), int(config["range_tvoc"][1])
    tvoc = st.number_input(
        label = "5.\tEnter tvoc Value",
        min_value = min_tvoc,
        max_value = max_tvoc,
        help = f"Value range from {min_tvoc} to {max_tvoc}"
    )

    min_co2, max_co2 = int(config["range_co2"][0]), int(config["range_co2"][1])
    co2 = st.number_input(
        label = "6.\tEnter co2 Value",
        min_value = min_co2,
        max_value = max_co2,
        help = f"Value range from {min_co2} to {max_co2}"
    )

    min_h2, max_h2 = int(config["range_raw_h2"][0]), int(config["range_raw_h2"][1])
    h2 = st.number_input(
        label = "7.\tEnter raw_h2 Value",
        min_value = min_h2,
        max_value = max_h2,
        help = f"Value range from {min_h2} to {max_h2}"
    )

    min_ethanol, max_ethanol = int(config["range_raw_ethanol"][0]), int(config["range_raw_ethanol"][1])
    ethanol = st.number_input(
        label = "8.\tEnter raw_ethanol Value",
        min_value = min_ethanol,
        max_value = max_ethanol,
        help = f"Value rage from {min_ethanol} to {max_ethanol}"
    )

    # Create button to submit the form.
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted.
    if submitted:
        # Create dict of all data in the form.
        raw_data = {
            "temperature" : temp,
            "humidity_pct": humid,
            "pressure": pressure,
            "pm10": pm10,
            "tvoc": tvoc,
            "co2": co2,
            "raw_h2": h2,
            "raw_ethanol": ethanol
        }

        # Create a loading animation while predicting.
        with st.spinner("Sending data to prediction server..."):
            res = requests.post("http://localhost:8080/predict", json=raw_data).json()

        # Parse the prediction.
        if res["error_msg"] != "":
            st.error(f"Error occurs while predicting: {res["error_msg"]}")
        else:
            if res["res"] != "FIRE NOT DETECTED.":
                st.error("WARNING: FIRE DETECTED!")
            else:
                st.success("FIRE NOT DETECTED.")