import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from geopy.geocoders import Nominatim
import numpy as np
from scipy import stats
from scipy.signal import detrend
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(page_title="Zillow Data Dashboard", layout="wide")

# Load data at app startup
try:
    data = load_all_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    data_loaded = False

# Set Mapbox token
px.set_mapbox_access_token(st.secrets["mapbox"]["token"])

# Sidebar for navigation
category = st.sidebar.selectbox(
    "Choose a Dashboard",
    ["Market Overview", "Market Analysis", "Market Activity", "Market Heatmap"]
)

if data_loaded:
    if category == "Market Heatmap":
        st.write("# 🗺️ National Market Investment Heatmap")
        
        # Calculate scores for all metros
        with st.spinner("Calculating investment scores for all metro areas..."):
            metro_scores_df = calculate_all_metro_scores()
        
        if not metro_scores_df.empty:
            # Debug information
            st.write("DataFrame Columns:", metro_scores_df.columns.tolist())
            st.write("DataFrame Head:", metro_scores_df.head())
            
            # Create the heatmap
            fig = px.scatter_mapbox(data_frame=metro_scores_df,
                                  lat='lat',
                                  lon='lon',
                                  color='investment_score',
                                  size='investment_score',
                                  hover_name='metro',
                                  hover_data=['investment_score', 'pr_ratio', 'price_trend', 'rent_trend'],
                                  color_continuous_scale='RdYlGn',
                                  size_max=25,
                                  zoom=3,
                                  center=dict(lat=39.8283, lon=-98.5795),  # Center of USA
                                  title='Investment Score Heatmap')
            
            fig.update_layout(mapbox_style='light')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No data available for the heatmap. Please check the coordinate retrieval process.")
            
    elif category == "Market Overview":
        # Get list of all states and metros, filtering out NaN values
        states = sorted(data['home_values']['StateName'].dropna().unique())
        
        # State selection
        selected_state = st.sidebar.selectbox(
            "Select State",
            states,
            index=states.index('California') if 'California' in states else 0
        )
        
        # Filter metros by selected state
        state_metros = sorted(data['home_values'][data['home_values']['StateName'] == selected_state]['RegionName'].dropna().unique())
        
        # Metro selection
        selected_metro = st.sidebar.selectbox(
            "Select Metro Area",
            state_metros
        )
        
        if selected_metro:
            st.write(f"# 📊 Advanced Market Analysis: {selected_metro}")
            
            # Get all the required data
            price_data = melt_data(data['home_values'][data['home_values']['RegionName'] == selected_metro])
            rental_data = melt_data(data['rentals'][data['rentals']['RegionName'] == selected_metro])
            inventory_data = melt_data(data['inventory'][data['inventory']['RegionName'] == selected_metro])
            new_listings_data = melt_data(data['new_listings'][data['new_listings']['RegionName'] == selected_metro])
            sold_above_data = melt_data(data['sold_above'][data['sold_above']['RegionName'] == selected_metro])
            
            # Calculate all metrics
            momentum = calculate_market_momentum(price_data, inventory_data, sold_above_data)
            pr_ratio = calculate_price_rent_ratio(price_data, rental_data)
            market_cycle = analyze_market_cycle(price_data)
            imbalance = calculate_market_imbalance(inventory_data, price_data, new_listings_data)
            volatility = calculate_volatility_metrics(price_data, sold_above_data)
            
            # Create layout
            st.write("## 🎯 Market Indicators")
            
            # First row - Market Momentum and Investment Score
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Market Momentum Score", f"{momentum:.1f}")
                st.plotly_chart(plot_market_pressure_gauge(imbalance), use_container_width=True)
                
            with col2:
                if pr_ratio:
                    st.metric("Investment Score", f"{pr_ratio['investment_score']:.1f}")
                    st.plotly_chart(plot_investment_score_gauge(pr_ratio), use_container_width=True)
            
            # Second row - Market Cycle and Risk Score
            col1, col2 = st.columns(2)
            with col1:
                if market_cycle:
                    st.write(f"### Market Cycle: {market_cycle['phase']}")
                    st.plotly_chart(plot_market_cycle_gauge(market_cycle), use_container_width=True)
                    
            with col2:
                if volatility:
                    st.write(f"### Market Stability")
                    st.plotly_chart(plot_risk_gauge(volatility), use_container_width=True)
            
            # Third row - Price/Rent Analysis
            if pr_ratio:
                st.write("## 📈 Price/Rent Analysis")
                pr_fig = px.line(pr_ratio['pr_series'], x='Date', y='pr_ratio',
                               title=f"Price-to-Rent Ratio Over Time - {selected_metro}")
                pr_fig.add_hline(y=pr_ratio['historical_mean'], line_dash="dash", 
                               annotation_text="Historical Average")
                st.plotly_chart(pr_fig, use_container_width=True)
            
            # Fourth row - Market Components
            if market_cycle and 'components' in market_cycle:
                st.write("## 🔄 Market Cycle Components")
                
                cycle_fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=("Trend", "Seasonal", "Residual")
                )
                
                cycle_fig.add_trace(
                    go.Scatter(y=market_cycle['components']['trend'], 
                             name="Trend"),
                    row=1, col=1
                )
                cycle_fig.add_trace(
                    go.Scatter(y=market_cycle['components']['seasonal'], 
                             name="Seasonal"),
                    row=2, col=1
                )
                cycle_fig.add_trace(
                    go.Scatter(y=market_cycle['components']['residual'], 
                             name="Residual"),
                    row=3, col=1
                )
                
                cycle_fig.update_layout(height=600, title_text=f"Market Trends - {selected_metro}")
                st.plotly_chart(cycle_fig, use_container_width=True)
                
    elif category == "Market Analysis":
        # Get list of all states and metros, filtering out NaN values
        states = sorted(data['home_values']['StateName'].dropna().unique())
        
        # State selection
        selected_state = st.sidebar.selectbox(
            "Select State",
            states,
            index=states.index('California') if 'California' in states else 0
        )
        
        # Filter metros by selected state
        state_metros = sorted(data['home_values'][data['home_values']['StateName'] == selected_state]['RegionName'].dropna().unique())
        
        # Metro selection
        selected_metro = st.sidebar.selectbox(
            "Select Metro Area",
            state_metros
        )
        
        if selected_metro:
            st.write(f"# 📈 Market Analysis Dashboard: {selected_metro}")
            
            # Home Values
            home_data = melt_data(data['home_values'][data['home_values']['RegionName'] == selected_metro])
            home_metrics = safe_calculate_metrics(home_data)
            
            # Rental Data
            rental_data = melt_data(data['rentals'][data['rentals']['RegionName'] == selected_metro])
            rental_metrics = safe_calculate_metrics(rental_data)
            
            # Market Activity
            inventory_data = melt_data(data['inventory'][data['inventory']['RegionName'] == selected_metro])
            list_price_data = melt_data(data['list_price'][data['list_price']['RegionName'] == selected_metro])
            sold_above_data = melt_data(data['sold_above'][data['sold_above']['RegionName'] == selected_metro])
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Median Home Value", f"${home_metrics['current_value']:,.0f}", 
                         f"{home_metrics['yoy_change']:.1f}% year over year")
            with col2:
                st.metric("Median Rent", f"${rental_metrics['current_value']:,.0f}", 
                         f"{rental_metrics['yoy_change']:.1f}% year over year")
            with col3:
                current_sold_above = safe_get_value(sold_above_data)
                yoy_sold_above = safe_get_value(sold_above_data, position=-13)
                st.metric("Homes Sold Above List", 
                         f"{current_sold_above:.1f}%",
                         f"{current_sold_above - yoy_sold_above:.1f}pp year over year")
            
            # Create visualizations if we have data
            if not all(df.empty for df in [home_data, rental_data, inventory_data, list_price_data]):
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Home Values", "Rental Prices", 
                                  "Available Inventory", "Median List Price")
                )
                
                # Add traces only if data exists
                if not home_data.empty:
                    fig.add_trace(
                        go.Scatter(x=home_data['Date'], y=home_data['Value'],
                                 name="Home Values"),
                        row=1, col=1
                    )
                
                if not rental_data.empty:
                    fig.add_trace(
                        go.Scatter(x=rental_data['Date'], y=rental_data['Value'],
                                 name="Rental Prices"),
                        row=1, col=2
                    )
                
                if not inventory_data.empty:
                    fig.add_trace(
                        go.Scatter(x=inventory_data['Date'], y=inventory_data['Value'],
                                 name="Inventory"),
                        row=2, col=1
                    )
                
                if not list_price_data.empty:
                    fig.add_trace(
                        go.Scatter(x=list_price_data['Date'], y=list_price_data['Value'],
                                 name="List Price"),
                        row=2, col=2
                    )
                
                fig.update_layout(height=800, title_text=f"Market Trends - {selected_metro}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for visualization")
                
    else:  # Market Activity
        # Get list of all states and metros, filtering out NaN values
        states = sorted(data['home_values']['StateName'].dropna().unique())
        
        # State selection
        selected_state = st.sidebar.selectbox(
            "Select State",
            states,
            index=states.index('California') if 'California' in states else 0
        )
        
        # Filter metros by selected state
        state_metros = sorted(data['home_values'][data['home_values']['StateName'] == selected_state]['RegionName'].dropna().unique())
        
        # Metro selection
        selected_metro = st.sidebar.selectbox(
            "Select Metro Area",
            state_metros
        )
        
        if selected_metro:
            st.write("## 📊 Market Activity")
            
            # Load and process market activity data
            inventory_data = melt_data(data['inventory'][data['inventory']['RegionName'] == selected_metro])
            new_listings_data = melt_data(data['new_listings'][data['new_listings']['RegionName'] == selected_metro])
            sold_above_data = melt_data(data['sold_above'][data['sold_above']['RegionName'] == selected_metro])
            list_price_data = melt_data(data['list_price'][data['list_price']['RegionName'] == selected_metro])
            
            # Display current metrics
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric("Available Inventory", 
                        f"{safe_get_value(inventory_data):,.0f}")
            with m2:
                st.metric("New Listings (Last Week)", 
                        f"{safe_get_value(new_listings_data):,.0f}")
            with m3:
                st.metric("Median List Price", 
                        f"${safe_get_value(list_price_data):,.0f}")
            with m4:
                st.metric("Sold Above List", 
                        f"{safe_get_value(sold_above_data):.1f}%")
            
            # Create tabs for different metrics
            tab1, tab2, tab3 = st.tabs(["📈 Inventory", "🏠 List Prices", "💰 Sales"])
            
            with tab1:
                if not inventory_data.empty:
                    fig = px.line(inventory_data.tail(52), x='Date', y='Value',
                                title=f"Available Inventory - {selected_metro}")
                    fig.update_layout(hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for visualization")
            
            with tab2:
                if not list_price_data.empty:
                    fig = px.line(list_price_data.tail(52), x='Date', y='Value',
                                title=f"Median List Price - {selected_metro}")
                    fig.update_layout(hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for visualization")
            
            with tab3:
                if not sold_above_data.empty:
                    fig = px.line(sold_above_data.tail(52), x='Date', y='Value',
                                title=f"Percentage Sold Above List - {selected_metro}")
                    fig.update_layout(hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for visualization")
else:
    st.error("Error loading data. Please check the data files and try again.")
