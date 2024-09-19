# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:46:27 2019
@author: PRATYUSH, Rahul, Somya, Abhay
"""

import numpy as np
import pandas as pd
from datetime import datetime
import random
import streamlit as st
from sklearn.tree import DecisionTreeRegressor

# Commodity data paths
commodity_dict = {
    "arhar": "static/Arhar.csv",
    "gram": "static/Gram.csv",
    "groundnut": "static/Groundnut.csv",
    "masoor": "static/Masoor.csv",
    "moong": "static/Moong.csv",
    "sugarcane": "static/Sugarcane.csv",
    "sunflower": "static/Sunflower.csv",
    "soyabean": "static/Soyabean.csv",
    "wheat": "static/Wheat.csv",
    "urad": "static/Urad.csv"
}

annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]

base = {
    "Arhar": 3200,
    "Gram": 2800,
    "Groundnut": 3700,
    "Masoor": 2800,
    "Moong": 3500,
    "Sugarcane": 2250,
    "Sunflower": 3700,
    "Soyabean": 2200,
    "Wheat": 1350,
    "Urad": 4300
}

commodity_list = []

class Commodity:
    def __init__(self, csv_name):
        self.name = csv_name
        dataset = pd.read_csv(csv_name)
        self.X = dataset.iloc[:, :-1].values
        self.Y = dataset.iloc[:, 3].values

        depth = random.randrange(7, 18)
        self.regressor = DecisionTreeRegressor(max_depth=depth)
        self.regressor.fit(self.X, self.Y)

    def getPredictedValue(self, value):
        if value[1] >= 2019:
            fsa = np.array(value).reshape(1, 3)
            return self.regressor.predict(fsa)[0]
        else:
            c = self.X[:, 0:2]
            x = []
            for i in c:
                x.append(i.tolist())
            fsa = [value[0], value[1]]
            ind = 0
            for i in range(0, len(x)):
                if x[i] == fsa:
                    ind = i
                    break
            return self.Y[ind]

    def getCropName(self):
        a = self.name.split('.')
        return a[0]

# Top 5 Winners Function
def TopFiveWinners():
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        current_month_prediction.append(current_predict)
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))

    sorted_change = sorted(change, reverse=True)
    to_send = []
    for j in range(5):
        perc, i = sorted_change[j]
        name = commodity_list[i].getCropName().split('/')[1]
        to_send.append([name, round((current_month_prediction[i] * base[name]) / 100, 2), round(perc, 2)])

    return to_send

# Top 5 Losers Function
def TopFiveLosers():
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        current_month_prediction.append(current_predict)
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))

    sorted_change = sorted(change)
    to_send = []
    for j in range(5):
        perc, i = sorted_change[j]
        name = commodity_list[i].getCropName().split('/')[1]
        to_send.append([name, round((current_month_prediction[i] * base[name]) / 100, 2), round(perc, 2)])

    return to_send

# 6 Months Forecast Helper
def SixMonthsForecastHelper(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 7):
        if current_month + i <= 12:
            month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
        else:
            month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[current_month + i - 13]))

    wpis = []
    for m, y, r in month_with_year:
        wpis.append(commodity.getPredictedValue([float(m), y, r]))

    crop_price = []
    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y, m, 1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i] * base[name.capitalize()]) / 100, 2)])

    return crop_price

# 12 Months Forecast Helper
def TwelveMonthsForecast(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 13):
        if current_month + i <= 12:
            month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
        else:
            month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[current_month + i - 13]))

    wpis = []
    for m, y, r in month_with_year:
        wpis.append(commodity.getPredictedValue([float(m), y, r]))

    crop_price = []
    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y, m, 1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i] * base[name.capitalize()]) / 100, 2)])

    max_val = max(crop_price, key=lambda x: x[1])
    min_val = min(crop_price, key=lambda x: x[1])

    return crop_price, max_val, min_val

# Initialize the commodity objects
commodity_list = [Commodity(commodity_dict[name]) for name in commodity_dict]

# Streamlit UI

# Sidebar for navigation
st.sidebar.title("Commodity Price Prediction")
st.sidebar.write("Select from the options below:")

# Sidebar buttons
if st.sidebar.button('Top 5 Winners'):
    winners = TopFiveWinners()
    st.write("### Top 5 Winners")
    st.table(pd.DataFrame(winners, columns=["Commodity", "Price", "Change (%)"]))

if st.sidebar.button('Top 5 Losers'):
    losers = TopFiveLosers()
    st.write("### Top 5 Losers")
    st.table(pd.DataFrame(losers, columns=["Commodity", "Price", "Change (%)"]))

# Dropdown for 6 and 12 Months Forecast
st.sidebar.header("Forecast Options")
option = st.sidebar.selectbox("Select an option:", ["6 Months Forecast", "12 Months Forecast"])

if option == "6 Months Forecast":
    crop_name = st.sidebar.selectbox("Select a crop:", list(commodity_dict.keys()))
    forecast = SixMonthsForecastHelper(crop_name)
    st.markdown(f"<h3 style='color: purple;'>6 Months Forecast for {crop_name.capitalize()}:</h3>", unsafe_allow_html=True)
    st.table(pd.DataFrame(forecast, columns=["Month", "Price"]))

elif option == "12 Months Forecast":
    crop_name = st.sidebar.selectbox("Select a crop:", list(commodity_dict.keys()))
    forecast, max_val, min_val = TwelveMonthsForecast(crop_name)
    st.markdown(f"<h3 style='color: purple;'>12 Months Forecast for {crop_name.capitalize()}:</h3>", unsafe_allow_html=True)
    st.table(pd.DataFrame(forecast, columns=["Month", "Price"]))
    st.write(f"### Max Price: {max_val[1]} in {max_val[0]}")
    st.write(f"### Min Price: {min_val[1]} in {min_val[0]}")
