import streamlit as st
import pandas as pd
import joblib
import sklearn
from datetime import date

# -----------------------------
# Load trained pipeline
# -----------------------------

saved = joblib.load("models/rossmann_pipeline.pkl")

model = saved["model"]
expected_columns = saved["columns"]

# -----------------------------
# App Title
# -----------------------------

st.title("Rossmann Sales Prediction")

# -----------------------------
# User Inputs
# -----------------------------

selected_date = st.date_input(
    "Select Date",
    value=date(2015, 7, 31),
    min_value=date(2013, 1, 1),
    max_value=date(2015, 7, 31)
)

customers = st.number_input(
    "Customers",
    min_value=1,
    max_value=8000,
    value=600
)

promo = st.selectbox("Promotion Active", [0, 1])

state_holiday = st.selectbox(
    "State Holiday",
    ["0", "a", "b", "c"]
)

school_holiday = st.selectbox(
    "School Holiday",
    [0, 1]
)

store_type = st.selectbox(
    "Store Type",
    ["a", "b", "c", "d"]
)

assortment = st.selectbox(
    "Assortment",
    ["a", "b", "c"]
)

competition_distance = st.number_input(
    "Competition Distance",
    min_value=20.0,
    max_value=80000.0,
    value=1200.0
)

competition_open_since_month = st.selectbox(
    "Competition Open Since Month",
    list(range(0, 13))
)

competition_open_since_year = st.number_input(
    "Competition Open Since Year",
    min_value=1900,
    max_value=2030,
    value=2008
)

promo2 = st.selectbox("Promo2", [0, 1])

if promo2 == 1:
    promo2_since_week = st.number_input(
        "Promo2 Since Week",
        min_value=1,
        max_value=52,
        value=13
    )

    promo2_since_year = st.number_input(
        "Promo2 Since Year",
        min_value=2009,
        max_value=2030,
        value=2012
    )

    promo_interval = st.selectbox(
        "Promo Interval",
        [
            "Jan,Apr,Jul,Oct",
            "Feb,May,Aug,Nov",
            "Mar,Jun,Sept,Dec"
        ]
    )
else:
    promo2_since_week = 0
    promo2_since_year = 0
    promo_interval = "None"

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Sales"):

    selected_date = pd.to_datetime(selected_date)

    year = selected_date.year
    month = selected_date.month
    day = selected_date.day

    # Rossmann DayOfWeek: Monday = 1, Sunday = 7
    day_of_week = selected_date.weekday() + 1

    # Python Weekday: Monday = 0, Sunday = 6
    weekday = selected_date.weekday()

    week_of_year = int(selected_date.isocalendar()[1])
    quarter = selected_date.quarter
    is_weekend = int(weekday >= 5)

    competition_age = max(
        year - competition_open_since_year,
        0
    )

    # Important: this matches the training logic
    promo_duration = max(
        year - promo2_since_year,
        0
    )

    is_holiday = int(state_holiday != "0")

    input_data = pd.DataFrame({
        "DayOfWeek": [day_of_week],
        "Customers": [customers],
        "Promo": [promo],
        "StateHoliday": [state_holiday],
        "SchoolHoliday": [school_holiday],
        "StoreType": [store_type],
        "Assortment": [assortment],
        "CompetitionDistance": [competition_distance],
        "CompetitionOpenSinceMonth": [competition_open_since_month],
        "CompetitionOpenSinceYear": [competition_open_since_year],
        "Promo2": [promo2],
        "Promo2SinceWeek": [promo2_since_week],
        "Promo2SinceYear": [promo2_since_year],
        "PromoInterval": [promo_interval],
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "Weekday": [weekday],
        "WeekOfYear": [week_of_year],
        "Quarter": [quarter],
        "IsWeekend": [is_weekend],
        "CompetitionAge": [competition_age],
        "PromoDuration": [promo_duration],
        "IsHoliday": [is_holiday]
    })

    input_data = input_data[expected_columns]


    prediction = model.predict(input_data)[0]

    if prediction < 0:
        prediction = 0

    st.success(f"Predicted Sales: {prediction:,.2f}")