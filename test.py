#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 14:39:53 2023

@author: hassan
"""

import streamlit as st
import currencyapicom
import time
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import hydralit_components as hc

st.set_page_config(layout="wide", page_title="Currency Analysis App")

@st.cache(allow_output_mutation=True)
def load_data(file_paths):
    dataframes = []
    for file_path in file_paths:
        df = pd.read_excel(file_path, parse_dates=['DATE_TIME'])
        dataframes.append(df)
    return dataframes

@st.cache(allow_output_mutation=True)
def clean_data(df, pair):
    # Your cleaning steps
    print(f'Processing {pair}')

@st.cache(allow_output_mutation=True)
def feature_engineering(df, pair):
    df['AVG_PRICE'] = (df['HIGH'] + df['LOW']) / 2
    df['VOLATILITY'] = df['HIGH'] - df['LOW']
    return df

# File paths of the 14 Excel files
file_paths = [
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/AUDUSD-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/EURCHF-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/EURJPY-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/EURUSD-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/USDCAD-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/USDCHF-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/USDJPY-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/USDAUD-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/CHFEUR-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/JPYEUR-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/USDEUR-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/CADUSD-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/CHFUSD-2000-2020-15m.xlsx",
    "/Users/hassan/Desktop/AUB/Masters/Capstone/data/JPYUSD-2000-2020-15m.xlsx"

]

    
# List of currency pairs
currency_pairs = ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF",
                  "USDJPY", "USDAUD", "CHFEUR", "JPYEUR", "USDEUR", "CADUSD",
                  "CHFUSD", "JPYUSD"]

# Load the data
dataframes = load_data(file_paths)

# Clean the data
for pair, df in zip(currency_pairs, dataframes):
    clean_data(df, pair)

# Perform feature engineering
for pair, df in zip(currency_pairs, dataframes):
    feature_engineering(df, pair)


def load_lottiefile(path):
    with open(path, "r") as f:
        return json.load(f)

# Lottie menu
menu_data = [
    {"label": "Home", "icon": "bi bi-house"},
    {"label": "Data Visualization", "icon": "bi bi-graph-up"},
    {"label": "Arbitrage Finder", "icon": "bi bi-search"}
]

#menu_id = hc.nav_bar(menu_definition=menu_data)

menu_id = st.selectbox('Menu', options=[x['label'] for x in menu_data])

# Home Page
if menu_id == "Home":
    st.title("Exploring Arbitrage Opportunities")
    st.write("""
    ---

    ###     By Hassan Hodroj

    #### For Master's in Business Analytics (MSBA), American University of Beirut (AUB)

    """)

    # Introduction
    st.markdown("""
    Welcome to this Streamlit app that aims to revolutionize the way you approach currency trading. 
    The app provides a comprehensive dashboard offering a multi-faceted view of currency behavior and intercurrency relationships. 
    Metrics like average price, volatility, open prices, and highest prices offer keen insights into both the stability and the risk associated with each currency pair.

    In addition, the app provides actionable insights by highlighting the best months and days to buy or sell a particular currency pair, 
    and displays a correlation matrix to reveal the potential intercurrency patterns. Most importantly, the app identifies arbitrage opportunities in real-time, 
    empowering traders to make quick, data-driven decisions for immediate profit.

    Dive in to explore, analyze, and seize profitable opportunities!
    """)
    
    
    
    # Load and display the Lottie animation
    lottie_file = load_lottiefile("/Users/hassan/Desktop/animation_lmdj278y.json")
    st_lottie(lottie_file, width=200, height=200)


# Data Visualization Page
elif menu_id == "Data Visualization":
    
    st.subheader('Visualizing Average Price and Volatility')
    
    # Assume dataframes is a list containing your data, and currency_pairs contains the names
    currency_pairs = ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF", "USDJPY"]

    # Sidebar for user input
    st.sidebar.header('Visualizing Average Price and Volatility Parameters')

    year_to_filter = st.sidebar.selectbox('Select Year', list(range(2000, 2020)))
    currency_to_filter = st.sidebar.selectbox('Select Currency Pair', currency_pairs)

    # Identify the selected currency pair's index
    idx = currency_pairs.index(currency_to_filter)

    # Get the relevant dataframe from your dataframes list
    selected_df = dataframes[idx]

    # Filter the data based on the selected year
    selected_df['YEAR'] = pd.to_datetime(selected_df['DATE_TIME']).dt.year
    filtered_df = selected_df[selected_df['YEAR'] == year_to_filter]

    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot AVG_PRICE
    axes[0].plot(filtered_df['DATE_TIME'], filtered_df['AVG_PRICE'], label='AVG_PRICE', color='blue')
    axes[0].set_title(f"AVG_PRICE for {currency_to_filter} ({year_to_filter})")
    axes[0].set_xlabel('Date Time')
    axes[0].set_ylabel('AVG_PRICE')
    axes[0].legend()

    # Plot VOLATILITY
    axes[1].plot(filtered_df['DATE_TIME'], filtered_df['VOLATILITY'], label='VOLATILITY', color='red')
    axes[1].set_title(f"VOLATILITY for {currency_to_filter} ({year_to_filter})")
    axes[1].set_xlabel('Date Time')
    axes[1].set_ylabel('VOLATILITY')
    axes[1].legend()

    # Adjust layout and show the plot in Streamlit
    plt.tight_layout()
    st.pyplot(fig)

    # In[10]:

    # Main Title
    st.subheader('Highest Price Reached')

    # Your list of dataframes and currency pairs
    currency_pairs = ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF", "USDJPY"]

    # User Input Sidebar
    st.sidebar.header('Highest Price Reached Parameters')
    currency_to_filter = st.sidebar.selectbox('Select Currency Pair', currency_pairs, key='unique_key_for_this_selectbox')

    # Index for selected dataframe
    idx = currency_pairs.index(currency_to_filter)

    # Relevant dataframe
    selected_df = dataframes[idx]

    # Process data
    selected_df['YEAR'] = pd.to_datetime(selected_df['DATE_TIME']).dt.year
    selected_df['MONTH'] = pd.to_datetime(selected_df['DATE_TIME']).dt.month
    monthly_avg = selected_df.groupby(['YEAR', 'MONTH'])['AVG_PRICE'].mean().reset_index()

    # Identify best and worst months
    best_sell_month = monthly_avg.loc[monthly_avg['AVG_PRICE'].idxmax()]
    best_buy_month = monthly_avg.loc[monthly_avg['AVG_PRICE'].idxmin()]

    # Display Results
    st.write(f"##### Best and Worst Trading Months for {currency_to_filter}")
    st.write(f"Best Month to SELL: {best_sell_month['MONTH']} in {best_sell_month['YEAR']} with avg price: {best_sell_month['AVG_PRICE']:.2f}")
    st.write(f"Best Month to BUY: {best_buy_month['MONTH']} in {best_buy_month['YEAR']} with avg price: {best_buy_month['AVG_PRICE']:.2f}")

    # In[ ]:

    # Provided data
    data = {
        'EURUSD': {'Best_month_to_SELL': 7.0, 'SELL_year': 2008.0, 'SELL_avg_price': 1.58, 'Best_month_to_BUY': 6.0, 'BUY_year': 2001.0, 'BUY_avg_price': 0.85},
        'USDJPY': {'Best_month_to_SELL': 2.0, 'SELL_year': 2002.0, 'SELL_avg_price': 133.57, 'Best_month_to_BUY': 10.0, 'BUY_year': 2011.0, 'BUY_avg_price': 76.66},
        'EURJPY': {'Best_month_to_SELL': 7.0, 'SELL_year': 2008.0, 'SELL_avg_price': 168.37, 'Best_month_to_BUY': 10.0, 'BUY_year': 2000.0, 'BUY_avg_price': 92.65},
        'USDCHF': {'Best_month_to_SELL': 6.0, 'SELL_year': 2001.0, 'SELL_avg_price': 1.78, 'Best_month_to_BUY': 8.0, 'BUY_year': 2011.0, 'BUY_avg_price': 0.78},
        'EURCHF': {'Best_month_to_SELL': 10.0, 'SELL_year': 2007.0, 'SELL_avg_price': 1.67, 'Best_month_to_BUY': 1.0, 'BUY_year': 2015.0, 'BUY_avg_price': 1.01},
        'AUDUSD': {'Best_month_to_SELL': 7.0, 'SELL_year': 2011.0, 'SELL_avg_price': 1.08, 'Best_month_to_BUY': 4.0, 'BUY_year': 2001.0, 'BUY_avg_price': 0.50},
        'USDCAD': {'Best_month_to_SELL': 1.0, 'SELL_year': 2002.0, 'SELL_avg_price': 1.60, 'Best_month_to_BUY': 7.0, 'BUY_year': 2011.0, 'BUY_avg_price': 0.96}
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    ind = np.arange(len(data))  # the x locations for the groups
    width = 0.35  # the width of the bars

    # Plotting
    sell_data = [info['Best_month_to_SELL'] for info in data.values()]
    buy_data = [info['Best_month_to_BUY'] for info in data.values()]

    sell_bars = ax.barh(ind, sell_data, width, color='r', label='Best Month to Sell')
    buy_bars = ax.barh(ind + width, buy_data, width, color='g', label='Best Month to Buy')

    # Annotations
    for i, bar in enumerate(sell_bars):
        pair = list(data.keys())[i]
        info = data[pair]
        ax.text(bar.get_width() - 1, bar.get_y() + bar.get_height()/2, f"Year: {info['SELL_year']}, Avg Price: {info['SELL_avg_price']}")
        
    for i, bar in enumerate(buy_bars):
        pair = list(data.keys())[i]
        info = data[pair]
        ax.text(bar.get_width() - 1, bar.get_y() + bar.get_height()/2, f"Year: {info['BUY_year']}, Avg Price: {info['BUY_avg_price']}")

    # Labels and Titles
    ax.set_xlabel('Months')
    ax.set_title('Highest and Lowest Price Reached by Currency Pairs')
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(data.keys())
    ax.legend()

    plt.show()


    # In[11]:
        
    # Title
    st.subheader('Best Month for a Pair to Buy and Sell')

    # Assume dataframes is your list of DataFrame objects.
    # currency_pairs is your list of currency pair names.
    currency_pairs = ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF", "USDJPY"]

    # Sidebar
    st.sidebar.header('Best Month for a Pair to Buy and Sell Parameters')
    selected_pair = st.sidebar.selectbox('Choose a currency pair', currency_pairs)

    # Index for selected DataFrame
    idx = currency_pairs.index(selected_pair)

    # Get the selected DataFrame
    selected_df = dataframes[idx]

    # Process DataFrame
    selected_df['YEAR'] = pd.to_datetime(selected_df['DATE_TIME']).dt.year
    selected_df['MONTH'] = pd.to_datetime(selected_df['DATE_TIME']).dt.month

    # Monthly Average
    monthly_avg = selected_df.groupby(['YEAR', 'MONTH'])['AVG_PRICE'].mean().reset_index()

    # Best and Worst Months
    best_sell_month_each_year = monthly_avg.loc[monthly_avg.groupby('YEAR')['AVG_PRICE'].idxmax()]['MONTH'].value_counts()
    best_buy_month_each_year = monthly_avg.loc[monthly_avg.groupby('YEAR')['AVG_PRICE'].idxmin()]['MONTH'].value_counts()

    # Overall best months
    best_sell_month_overall = best_sell_month_each_year.idxmax()
    best_buy_month_overall = best_buy_month_each_year.idxmax()

    # Display the results
    st.write(f"##### Results for {selected_pair}")
    st.write(f"Best month to SELL on average is {best_sell_month_overall} (appeared {best_sell_month_each_year.max()} times)")
    st.write(f"Best month to BUY on average is {best_buy_month_overall} (appeared {best_buy_month_each_year.max()} times)")


    # In[ ]:
   
    # The data you provided
    results = {
        'EURUSD': {'best_month_to_sell': 1, 'times_best_to_sell': 5, 'best_month_to_buy': 2, 'times_best_to_buy': 4},
        'USDJPY': {'best_month_to_sell': 12, 'times_best_to_sell': 6, 'best_month_to_buy': 1, 'times_best_to_buy': 5},
        'EURJPY': {'best_month_to_sell': 12, 'times_best_to_sell': 9, 'best_month_to_buy': 12, 'times_best_to_buy': 4},
        'USDCHF': {'best_month_to_sell': 11, 'times_best_to_sell': 5, 'best_month_to_buy': 12, 'times_best_to_buy': 5},
        'EURCHF': {'best_month_to_sell': 1, 'times_best_to_sell': 5, 'best_month_to_buy': 1, 'times_best_to_buy': 3},
        'AUDUSD': {'best_month_to_sell': 1, 'times_best_to_sell': 7, 'best_month_to_buy': 11, 'times_best_to_buy': 3},
        'USDCAD': {'best_month_to_sell': 1, 'times_best_to_sell': 5, 'best_month_to_buy': 1, 'times_best_to_buy': 6},
    }

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(15, 8))

    # Create an index for each tick position
    ind = np.arange(len(results))

    # Width of a bar
    width = 0.35       

    # Plotting
    best_sell_months = [data['best_month_to_sell'] for data in results.values()]
    best_buy_months = [data['best_month_to_buy'] for data in results.values()]

    sell_bar = ax.barh(ind, best_sell_months, width, color='red', label='Best Month to Sell')
    buy_bar = ax.barh(ind + width, best_buy_months, width, color='green', label='Best Month to Buy')

    # Describe the data
    ax.set_xlabel('Months of the Year')
    ax.set_title('Best Month to Buy and Sell by Currency Pair')
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(results.keys())
    ax.legend()

    plt.show()


    # In[12]:

    # Assume dataframes is your list of DataFrame objects.
    # You will need to load these dataframes into your script.

    # Title
    st.subheader('Best Day for a Pair to Buy and Sell')

    # Sidebar for user input
    st.sidebar.header('Best Day for a Pair to Buy and Sell')
    selected_pair = st.sidebar.selectbox('Choose a currency pair', ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF","USDJPY"], key='unique_key_for_selectbox')

    # Find index of selected currency pair
    idx = ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF","USDJPY"].index(selected_pair)

    # Select the DataFrame for the chosen currency pair
    selected_df = dataframes[idx]

    # Extract year, month, and day from DATE_TIME column
    selected_df['YEAR'] = pd.to_datetime(selected_df['DATE_TIME']).dt.year
    selected_df['MONTH'] = pd.to_datetime(selected_df['DATE_TIME']).dt.month
    selected_df['DAY'] = pd.to_datetime(selected_df['DATE_TIME']).dt.day

    # Group by YEAR, MONTH, and DAY and calculate the mean for each day in each month
    daily_avg = selected_df.groupby(['YEAR', 'MONTH', 'DAY'])['AVG_PRICE'].mean().reset_index()

    # Find the best and worst days for each month
    best_sell_day_each_month = daily_avg.loc[daily_avg.groupby(['YEAR', 'MONTH'])['AVG_PRICE'].idxmax()]['DAY'].value_counts()
    best_buy_day_each_month = daily_avg.loc[daily_avg.groupby(['YEAR', 'MONTH'])['AVG_PRICE'].idxmin()]['DAY'].value_counts()

    # Overall best days
    best_sell_day_overall = best_sell_day_each_month.idxmax()
    best_buy_day_overall = best_buy_day_each_month.idxmax()

    # Display results
    st.write(f"##### Results for {selected_pair}")
    st.write(f"Best day to SELL on average is {best_sell_day_overall} (appeared {best_sell_day_each_month.max()} times)")
    st.write(f"Best day to BUY on average is {best_buy_day_overall} (appeared {best_buy_day_each_month.max()} times)")


    # In[ ]:

    # The data
    results = {
        'EURUSD': {'best_day_to_sell': 1, 'times_best_to_sell': 20, 'best_day_to_buy': 1, 'times_best_to_buy': 23},
        'USDJPY': {'best_day_to_sell': 1, 'times_best_to_sell': 19, 'best_day_to_buy': 1, 'times_best_to_buy': 24},
        'EURJPY': {'best_day_to_sell': 1, 'times_best_to_sell': 25, 'best_day_to_buy': 1, 'times_best_to_buy': 17},
        'USDCHF': {'best_day_to_sell': 1, 'times_best_to_sell': 21, 'best_day_to_buy': 1, 'times_best_to_buy': 20},
        'EURCHF': {'best_day_to_sell': 30, 'times_best_to_sell': 19, 'best_day_to_buy': 1, 'times_best_to_buy': 23},
        'AUDUSD': {'best_day_to_sell': 1, 'times_best_to_sell': 22, 'best_day_to_buy': 1, 'times_best_to_buy': 30},
        'USDCAD': {'best_day_to_sell': 1, 'times_best_to_sell': 24, 'best_day_to_buy': 1, 'times_best_to_buy': 20},
    }

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(15, 8))

    # Create an index for each tick position
    ind = np.arange(len(results))

    # Width of a bar 
    width = 0.35       

    # Plotting
    best_sell_days = [data['best_day_to_sell'] for data in results.values()]
    best_buy_days = [data['best_day_to_buy'] for data in results.values()]

    sell_bar = ax.barh(ind, best_sell_days, width, color='red', label='Best Day to Sell')
    buy_bar = ax.barh(ind + width, best_buy_days, width, color='green', label='Best Day to Buy')

    # Describe the data
    ax.set_xlabel('Days of the Month')
    ax.set_title('Best Day to Buy and Sell by Currency Pair')
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(results.keys())
    ax.legend()

    plt.show()


    # In[14]:


    #taking too long to run this is visualizing all years

    # Title of the Streamlit app
    st.subheader("Currency Pairs Visualization")

    # Sidebar to select the currency pair
    st.sidebar.header("Currency Pairs Visualization")
    selected_pair = st.sidebar.selectbox('Choose a currency pair', ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF","USDJPY"], key='unique_key_for_selectbox_1')


    # Get the index of the selected currency pair
    idx = ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF", "USDJPY"].index(selected_pair)

    # Initialize Matplotlib figure and axis
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Plotting AVG_PRICE
    axes[0].plot(selected_df['DATE_TIME'], selected_df['AVG_PRICE'], label='AVG_PRICE', color='blue')
    axes[0].set_title(f"AVG_PRICE for {selected_pair}")
    axes[0].set_xlabel('Date Time')
    axes[0].set_ylabel('AVG_PRICE')
    axes[0].legend()

    # Plotting VOLATILITY
    axes[1].plot(selected_df['DATE_TIME'], selected_df['VOLATILITY'], label='VOLATILITY', color='red')
    axes[1].set_title(f"VOLATILITY for {selected_pair}")
    axes[1].set_xlabel('Date Time')
    axes[1].set_ylabel('VOLATILITY')
    axes[1].legend()

    # Display Matplotlib chart in Streamlit
    st.pyplot(fig)


    # In[23]:


    #!pip install --upgrade numpy scipy seaborn


    # In[27]:
   
    # Title of the Streamlit app
    st.subheader("Currency Pairs Analysis")

    # Sidebar to select the currency pair and year
    st.sidebar.header("Currency Pairs Analysis")
    selected_pair = st.sidebar.selectbox('Choose a currency pair', ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF","USDJPY"], key='unique_key_for_selectbox_2')
    selected_year = st.sidebar.slider("Select a Year", min_value=2000, max_value=2020)

    # Your dataframes list should already be defined
    # dataframes = [...]

    # Get the index of the selected currency pair
    idx = ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF", "USDJPY"].index(selected_pair)
    selected_df = dataframes[idx]

    # Convert the string DATE_TIME to datetime format
    selected_df['DATE_TIME'] = pd.to_datetime(selected_df['DATE_TIME'], format="%Y.%m.%d %H:%M:%S")

    # Filter DataFrame for the selected year
    filtered_df = selected_df[selected_df['DATE_TIME'].dt.year == selected_year]

    # Line plot for OPEN prices
    st.write(f"#### Line Plot for OPEN Prices of {selected_pair} in {selected_year}")
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.plot(filtered_df['DATE_TIME'], filtered_df['OPEN'])
    plt.title(f' OPEN Prices for {selected_pair} in {selected_year}')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("#### Correlation Heatmap for OPEN Prices")

    # Initialize an empty DataFrame to store OPEN prices
    open_prices = pd.DataFrame()

    # Populate the DataFrame
    for pair in ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF", "USDJPY"]:
        idx = ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF", "USDJPY"].index(pair)
        open_prices[pair] = dataframes[idx]['OPEN']

    # Calculate the correlation matrix
    correlation = open_prices.corr()

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between OPEN Prices of Currency Pairs')
    plt.tight_layout()
    st.pyplot(fig)
    
   
    @st.cache(allow_output_mutation=True)
    def calculate_arbitrage_opportunities(dataframes, selected_path, selected_years):
        pathway = selected_path.split(", ")
        currency_pairs = ["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF", "USDJPY", "USDAUD", "CHFEUR", "JPYEUR", "USDEUR", "CADUSD", "CHFUSD", "JPYUSD"]
        start_year, end_year = selected_years
        results = []
        capital = 1_000_000  # Starting capital
        
        for pair in currency_pairs:
            idx = currency_pairs.index(pair)
            dataframes[idx] = dataframes[idx][dataframes[idx]['DATE_TIME'].dt.year.between(start_year, end_year)]

        common_timestamps = set(dataframes[currency_pairs.index(pathway[0])]['DATE_TIME'])
        for pair in pathway[1:]:
            common_timestamps.intersection_update(set(dataframes[currency_pairs.index(pair)]['DATE_TIME']))

        for timestamp in common_timestamps:
            product = 1
            for pair in pathway:
                idx = currency_pairs.index(pair)
                filtered_data = dataframes[idx][dataframes[idx]['DATE_TIME'] == timestamp]['OPEN'].values
                if filtered_data.size == 0:
                    break
                rate = filtered_data[0]
                product *= rate
            else:
                if product > 1:
                    capital *= product
                results.append((timestamp, product, capital))

        df_results = pd.DataFrame(results, columns=['DATE_TIME', 'ARBITRAGE_PRODUCT', 'CAPITAL_AFTER_TRADE'])
        return df_results

    # Streamlit App
    st.write("#### Arbitrage Opportunities in Historical Currency Rates")

    # Sidebar
    st.sidebar.header("Choose an Arbitrage Pathway")
    selected_path = st.sidebar.selectbox(
        'Available Pathways', 
        [
            "USDEUR, EURJPY, JPYUSD",
            "USDEUR, EURCHF, CHFUSD",
            "USDCHF, CHFEUR, EURUSD",
            "USDJPY, JPYEUR, EURCHF, CHFUSD",
            "USDAUD, AUDUSD, USDJPY, JPYUSD",
            "USDJPY, JPYEUR, EURCHF, CHFUSD",
            "USDJPY, JPYEUR, EURUSD, USDCHF, CHFUSD"
        ]
    )
    selected_years = st.sidebar.slider("Choose Year Range", 2000, 2020, (2000, 2010))

    # Assume that dataframes are loaded and available as a list
    # dataframes = [...]

    # Get cached data
    df_results = calculate_arbitrage_opportunities(dataframes, selected_path, selected_years)
    arbitrage_opportunities = df_results[df_results['ARBITRAGE_PRODUCT'] > 1]
    st.write(arbitrage_opportunities)

    # Visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    timestamps = arbitrage_opportunities["DATE_TIME"].values
    products = arbitrage_opportunities["ARBITRAGE_PRODUCT"].values
    ax.scatter(timestamps, products, color='red', marker='o')
    ax.axhline(y=1, color='blue', linestyle='--')

    selected_pathway_str = ', '.join(selected_path.split(", "))
    ax.set_title(f"Arbitrage Opportunities of pathway ({selected_pathway_str}) between {selected_years[0]} and {selected_years[1]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Arbitrage Product")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)






    


# In[31]:

#import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd


# List of currency pairs
#currency_pairs=["AUDUSD", "EURCHF", "EURJPY", "EURUSD", "USDCAD", "USDCHF","USDJPY"]

# Iterate over each currency pair dataset
#for pair in currency_pairs: 
#    idx=currency_pairs.index(pair)
#    selected_df = dataframes[idx]
    
    # Convert the string DATE_TIME to datetime format
#    selected_df['DATE_TIME'] = pd.to_datetime(selected_df['DATE_TIME'], format="%Y.%m.%d %H:%M:%S")
    
    # Filter for the year 2008
#    selected_df = selected_df[selected_df['DATE_TIME'].dt.year == 2008]
    

# Gather all volatility data for boxplot
#volatility_data = []
#pair_labels = []

# Get the smaller length among dataframes and currency_pairs
#min_length = min(len(dataframes), len(currency_pairs))

#for idx in range(min_length):
#    df = dataframes[idx]
#    pair = currency_pairs[idx]
#    volatility_data.append(df['VOLATILITY'].values)
#    pair_labels.append(pair)

# Create the boxplot
#plt.figure(figsize=(15, 8))
#sns.boxplot(data=volatility_data)
#plt.xticks(ticks=range(len(pair_labels)), labels=pair_labels)
#plt.ylabel('Volatility')
#plt.title('Volatility Distribution Across Currency Pairs in 2009')
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.tight_layout()
#plt.show()
    
     



# Data Visualization Page
elif menu_id == "Arbitrage Finder":
    st.subheader("Arbitrage Opportunity Finder")

    client = currencyapicom.Client('cur_live_DgMqHoLrdJB1dELBTjmyZXFRMEBtfKc5jgRWujVG')
    
    # Sidebar
    st.sidebar.header("Choose an Arbitrage Pathway")
    selected_path = st.sidebar.selectbox(
        'Available Pathways', 
        [
            "USDEUR, EURJPY, JPYUSD",
            "USDCAD, CADJPY, JPYUSD",
            "USDAUD, AUDJPY, JPYUSD",
            "USDAUD, AUDCAD, CADUSD",
            "USDJPY, JPYEUR, EURCHF, CHFUSD"
        ]
    )

    # Function to fetch rates for a given base currency
    def fetch_rates_for_base(base, currencies):
        result = client.latest(base, currencies=currencies)
        if result:
            return result['data']
        else:
            return None

    # Function to find arbitrage opportunities
    def find_arbitrage_opportunities(pathway):
        pathway = pathway.split(", ")
        all_rates = {}
        
        for currency in ['USD', 'EUR', 'JPY', 'CHF', 'CAD', 'AUD']:
            all_rates[currency] = fetch_rates_for_base(currency, ['EUR', 'USD', 'JPY', 'CHF', 'CAD', 'AUD'])

        amount = 1000000  # Starting amount in the currency of the first element in the pathway

        for p in pathway:
            base, target = p[:3], p[3:]
            if all_rates.get(base) and all_rates.get(target):
                rate = all_rates[base][target]['value']
                amount *= rate

        st.write(f"Final amount after following the pathway {', '.join(pathway)}: {amount:.2f}")
        
        if amount > 1000000:
            st.write(f"Arbitrage opportunity found! Starting with 1,000,000, you can end up with {amount:.2f}")
        else:
            st.write("No arbitrage opportunity found.")

    find_arbitrage_opportunities(selected_path)
    
    # Load and display the Lottie animation
    lottie_file = load_lottiefile("/Users/hassan/Desktop/animation_lmdkbh9e.json")
    st_lottie(lottie_file, width=200, height=200)


      