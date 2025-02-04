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
import re
import json
import os

# Configure the page
st.set_page_config(page_title="Zillow Data Dashboard", layout="wide")

# Load data at app startup
@st.cache_data
def load_all_data():
    """Load all available Zillow data files"""
    try:
        data = {}
        # Load each file with error handling
        files = {
            'home_values': "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
            'rentals': "Metro_zori_uc_sfrcondomfr_sm_month.csv",
            'inventory': "Metro_invt_fs_uc_sfrcondo_sm_week.csv",
            'list_price': "Metro_mlp_uc_sfrcondo_sm_week.csv",
            'new_listings': "Metro_new_listings_uc_sfrcondo_week.csv",
            'sold_above': "Metro_pct_sold_above_list_uc_sfrcondo_week.csv"
        }
        
        for key, filename in files.items():
            try:
                df = pd.read_csv(filename)
                # Ensure RegionName and StateName columns exist and are string type
                if 'RegionName' in df.columns:
                    df['RegionName'] = df['RegionName'].astype(str)
                if 'StateName' in df.columns:
                    df['StateName'] = df['StateName'].astype(str)
                data[key] = df
            except Exception as e:
                st.warning(f"Could not load {filename}: {str(e)}")
                # Provide empty DataFrame with required columns
                data[key] = pd.DataFrame(columns=['RegionName', 'StateName', 'Value'])
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

try:
    data = load_all_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.error("Error loading data. Please check the data files and try again.")
    data_loaded = False

# Set Mapbox token
px.set_mapbox_access_token(st.secrets["mapbox"]["token"])

# Common metro coordinates to avoid API calls
METRO_COORDINATES = {
    "United States": (39.8283, -98.5795),  # Geographic center of the contiguous US
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Dallas": (32.7767, -96.7970),
    "Houston": (29.7604, -95.3698),
    "Washington": (38.9072, -77.0369),
    "Philadelphia": (39.9526, -75.1652),
    "Miami": (25.7617, -80.1918),
    "Atlanta": (33.7490, -84.3880),
    "Boston": (42.3601, -71.0589),
    "Phoenix": (33.4484, -112.0740),
    "San Francisco": (37.7749, -122.4194),
    "Riverside": (33.9806, -117.3755),
    "Detroit": (42.3314, -83.0458),
    "Seattle": (47.6062, -122.3321),
    "Minneapolis": (44.9778, -93.2650),
    "San Diego": (32.7157, -117.1611),
    "Tampa": (27.9506, -82.4572),
    "Denver": (39.7392, -104.9903),
    "Baltimore": (39.2904, -76.6122),
    "St. Louis": (38.6270, -90.1994),
    "Charlotte": (35.2271, -80.8431),
    "Orlando": (28.5383, -81.3792),
    "San Antonio": (29.4241, -98.4936),
    "Portland": (45.5155, -122.6789),
    "Sacramento": (38.5816, -121.4944),
    "Pittsburgh": (40.4406, -79.9959),
    "Las Vegas": (36.1699, -115.1398),
    "Austin": (30.2672, -97.7431),
    "Cincinnati": (39.1031, -84.5120),
    "Kansas City": (39.0997, -94.5786),
    "Columbus": (39.9612, -82.9988),
    "Indianapolis": (39.7684, -86.1581),
    "Cleveland": (41.4993, -81.6944),
    "Nashville": (36.1627, -86.7816),
    "Virginia Beach": (36.8529, -75.9780),
    "Providence": (41.8240, -71.4128),
    "Milwaukee": (43.0389, -87.9065),
    "Jacksonville": (30.3322, -81.6557),
    "Oklahoma City": (35.4676, -97.5164),
    "Raleigh": (35.7796, -78.6382),
    "Memphis": (35.1495, -90.0490),
    "Richmond": (37.5407, -77.4360),
    "Buffalo": (42.8864, -78.8784),
    "Salt Lake City": (40.7608, -111.8910),
    "Rochester": (43.1566, -77.6088),
    "Grand Rapids": (42.9634, -85.6681),
    "Tucson": (32.2226, -110.9747),
    "Urban Honolulu": (21.3069, -157.8583),
    "Fresno": (36.7378, -119.7871),
    "Worcester": (42.2626, -71.8023),
    "Albuquerque": (35.0844, -106.6504),
    "Omaha": (41.2565, -95.9345),
    "Albany": (42.6526, -73.7562),
    "New Orleans": (29.9511, -90.0715),
    "Bakersfield": (35.3733, -119.0187),
    "Knoxville": (35.9606, -83.9207),
    "Ogden": (41.2230, -111.9738),
    "Allentown": (40.6084, -75.4902),
    "Baton Rouge": (30.4515, -91.1871),
    "McAllen": (26.2034, -98.2300),
    "Dayton": (39.7589, -84.1916),
    "Columbia": (34.0007, -81.0348),
    "Greensboro": (36.0726, -79.7920),
    "Akron": (41.0814, -81.5190),
    "Little Rock": (34.7465, -92.2896),
    "Stockton": (37.9577, -121.2908),
    "Colorado Springs": (38.8339, -104.8214),
    "Charleston": (32.7765, -79.9311),
    "Syracuse": (43.0481, -76.1474),
    "Winston-Salem": (36.0999, -80.2442),
    "Greenville": (34.8526, -82.3940),
    "Wichita": (37.6872, -97.3301),
    "Cape Coral": (26.5629, -81.9495),
    "Boise City": (43.6150, -116.2023),
    "Springfield": (42.1015, -72.5898),
    "Toledo": (41.6639, -83.5552),
    "Lakeland": (28.0395, -81.9498),
    "Madison": (43.0731, -89.4012)
}

# Load cached coordinates from file if it exists
CACHE_FILE = "metro_coordinates_cache.json"
CACHED_COORDINATES = {}

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, 'r') as f:
            CACHED_COORDINATES = json.load(f)
    except Exception:
        pass

def save_coordinates_cache():
    """Save the coordinates cache to file"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(CACHED_COORDINATES, f)
    except Exception:
        pass

def clean_metro_name(metro):
    """Clean metro name by removing state and other info"""
    # Skip cleaning for "United States"
    if metro == "United States":
        return metro
        
    # Handle cases where state is duplicated (e.g., "Los Angeles, CA, CA")
    metro = re.sub(r',\s*([A-Z]{2}),\s*\1', r', \1', metro)
    
    # Remove any state abbreviations (e.g., ", CA", ", NY")
    metro = re.sub(r',\s*[A-Z]{2}.*$', '', metro)
    
    # Remove any other suffixes like "Metro Area", "County", etc.
    metro = re.sub(r'\s*(Metro.*|County|City|Area).*$', '', metro, flags=re.IGNORECASE)
    
    return metro.strip()

@st.cache_data
def get_metro_coordinates(metro, state):
    """Get coordinates for a metro area using hardcoded values or Nominatim"""
    try:
        # Skip geocoding for "United States"
        if metro == "United States":
            return METRO_COORDINATES["United States"]
            
        # Clean the metro name to match our hardcoded values
        metro_key = clean_metro_name(metro)
        
        # First check if we have hardcoded coordinates
        if metro_key in METRO_COORDINATES:
            return METRO_COORDINATES[metro_key]
            
        # Check if we have cached coordinates
        cache_key = f"{metro_key}_{state}" if state else metro_key
        if cache_key in CACHED_COORDINATES:
            return tuple(CACHED_COORDINATES[cache_key])
            
        # Clean up state (remove duplicates and handle nan)
        if pd.isna(state):
            state = ""
        else:
            state = re.sub(r'([A-Z]{2}),\s*\1', r'\1', state)
            state = state.strip()
        
        # If not found in hardcoded values or cache, try geocoding
        geolocator = Nominatim(user_agent="zillow_dashboard")
        
        # Try different query formats
        queries = [
            f"{metro_key}, {state}, USA" if state else f"{metro_key}, USA",  # Full format
            f"{metro_key}, USA",  # Just city and country
            metro_key  # Just city
        ]
        
        for query in queries:
            try:
                location = geolocator.geocode(query, timeout=5)
                if location:
                    # Cache the coordinates
                    CACHED_COORDINATES[cache_key] = [location.latitude, location.longitude]
                    save_coordinates_cache()
                    return location.latitude, location.longitude
            except Exception:
                continue
        
        st.warning(f"Could not find coordinates for {metro_key}, {state}")
        return None
        
    except Exception as e:
        st.warning(f"Error getting coordinates for {metro}, {state}: {str(e)}")
        return None

def melt_data(df, id_cols=None):
    """Convert wide format data to long format"""
    try:
        if df.empty:
            return pd.DataFrame(columns=['Date', 'Value'])
            
        if id_cols is None:
            id_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
            # Only use columns that exist in the DataFrame
            id_cols = [col for col in id_cols if col in df.columns]
        
        # Get value columns (all columns not in id_cols)
        value_vars = [col for col in df.columns if col not in id_cols]
        
        if not value_vars:
            return pd.DataFrame(columns=['Date', 'Value'])
        
        melted = df.melt(
            id_vars=id_cols,
            value_vars=value_vars,
            var_name='Date',
            value_name='Value'
        )
        
        # Convert date strings to datetime, with error handling
        try:
            melted['Date'] = pd.to_datetime(melted['Date'])
        except:
            melted['Date'] = pd.to_datetime('today')
            
        return melted
    except Exception as e:
        st.warning(f"Error melting data: {str(e)}")
        return pd.DataFrame(columns=['Date', 'Value'])

def safe_get_value(df, col='Value', position=-1, default=0):
    """Safely get a value from a DataFrame, returning default if not found"""
    try:
        if df.empty:
            return default
        return df[col].iloc[position]
    except (IndexError, KeyError):
        return default

def safe_calculate_metrics(df, value_col='Value'):
    """Calculate metrics with error handling"""
    try:
        if df.empty:
            return {
                'current_value': 0,
                'yoy_change': 0,
                'avg_monthly_change': 0
            }
        
        current_value = df[value_col].iloc[-1]
        year_ago_value = df[value_col].iloc[-13] if len(df) > 13 else df[value_col].iloc[0]
        yoy_change = ((current_value - year_ago_value) / year_ago_value) * 100
        monthly_changes = df[value_col].pct_change() * 100
        avg_monthly_change = monthly_changes.tail(12).mean()
        
        return {
            'current_value': current_value,
            'yoy_change': yoy_change,
            'avg_monthly_change': avg_monthly_change
        }
    except (IndexError, KeyError):
        return {
            'current_value': 0,
            'yoy_change': 0,
            'avg_monthly_change': 0
        }

def calculate_market_momentum(price_data, inventory_data, sold_above_data):
    """Calculate market momentum score (0-100)"""
    try:
        scores = []
        
        # Price appreciation score (0-25)
        if not price_data.empty:
            yoy_change = ((price_data['Value'].iloc[-1] - price_data['Value'].iloc[-13]) / 
                         price_data['Value'].iloc[-13]) * 100
            price_score = min(max(yoy_change + 10, 0), 25)  # Scale and cap
            scores.append(price_score)
        
        # Inventory change score (0-25)
        if not inventory_data.empty:
            inv_change = ((inventory_data['Value'].iloc[-1] - inventory_data['Value'].iloc[-13]) / 
                         inventory_data['Value'].iloc[-13]) * 100
            inv_score = min(max(25 - inv_change, 0), 25)  # Inverse scale (lower inventory = higher score)
            scores.append(inv_score)
        
        # Sold above list score (0-25)
        if not sold_above_data.empty:
            sold_above = sold_above_data['Value'].iloc[-1]
            sold_score = min(max(sold_above, 0), 25)
            scores.append(sold_score)
        
        # Market velocity score (0-25)
        if not price_data.empty:
            recent_changes = price_data['Value'].pct_change().tail(6).mean() * 100
            velocity_score = min(max(recent_changes * 5 + 12.5, 0), 25)  # Scale and center
            scores.append(velocity_score)
        
        if scores:
            return sum(scores) / len(scores) * (100 / 25)  # Normalize to 0-100
        return 50  # Default neutral score
    except Exception as e:
        st.warning(f"Error calculating market momentum: {str(e)}")
        return 50

def calculate_price_rent_ratio(price_data, rental_data):
    """Calculate price-to-rent ratio and investment metrics"""
    try:
        if price_data.empty or rental_data.empty:
            return None
            
        # Calculate monthly P/R ratio
        merged = pd.merge(price_data, rental_data, on='Date', suffixes=('_price', '_rent'))
        merged['pr_ratio'] = merged['Value_price'] / (merged['Value_rent'] * 12)  # Annual rent
        
        # Calculate historical metrics
        current_ratio = merged['pr_ratio'].iloc[-1]
        mean_ratio = merged['pr_ratio'].mean()
        std_ratio = merged['pr_ratio'].std()
        z_score = (current_ratio - mean_ratio) / std_ratio
        
        # Calculate trend scores
        price_trend = merged['Value_price'].pct_change(12).iloc[-1] * 100
        rent_trend = merged['Value_rent'].pct_change(12).iloc[-1] * 100
        
        # Investment score (0-100)
        # Lower P/R ratio, higher rent growth = better investment
        pr_score = max(0, min(50 - z_score * 10, 50))  # P/R component
        trend_score = max(0, min(rent_trend * 2, 50))  # Rent trend component
        investment_score = pr_score + trend_score
        
        return {
            'current_ratio': current_ratio,
            'historical_mean': mean_ratio,
            'z_score': z_score,
            'price_trend': price_trend,
            'rent_trend': rent_trend,
            'investment_score': investment_score,
            'pr_series': merged[['Date', 'pr_ratio']].copy()
        }
    except Exception as e:
        st.warning(f"Error calculating P/R ratio: {str(e)}")
        return None

def analyze_market_cycle(price_data, periods=12):
    """Analyze market cycle position using time series decomposition"""
    try:
        if price_data.empty or len(price_data) < periods * 2:
            return None
            
        # Resample to monthly frequency and handle missing values
        monthly_data = price_data.set_index('Date').resample('M')['Value'].mean()
        
        # Interpolate missing values
        monthly_data = monthly_data.interpolate(method='cubic', limit_direction='both')
        
        # Need at least 2 full years of data after interpolation
        if len(monthly_data.dropna()) < periods * 2:
            return None
            
        # Ensure we have enough non-NaN values
        if monthly_data.isna().sum() > len(monthly_data) * 0.3:  # If more than 30% missing
            return None
            
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(monthly_data, period=periods, extrapolate_trend='freq')
            
            # Handle potential NaN values in components
            trend = pd.Series(decomposition.trend).interpolate(method='linear')
            seasonal = pd.Series(decomposition.seasonal).interpolate(method='linear')
            residual = pd.Series(decomposition.resid).interpolate(method='linear')
            
            # Determine cycle position
            recent_trend = trend.diff().tail(6).mean()
            recent_residual = residual.tail(6).mean()
            
            # Classify market phase
            if recent_trend > 0:
                if recent_residual > 0:
                    phase = "Expansion"
                    angle = 45
                else:
                    phase = "Peak"
                    angle = 135
            else:
                if recent_residual < 0:
                    phase = "Contraction"
                    angle = 225
                else:
                    phase = "Trough"
                    angle = 315
            
            # Calculate confidence (0-100)
            # Lower confidence if we had to interpolate a lot
            missing_penalty = (monthly_data.isna().sum() / len(monthly_data)) * 25
            trend_strength = abs(recent_trend) / trend.std()
            residual_strength = abs(recent_residual) / residual.std()
            confidence = max(0, min((trend_strength + residual_strength) * 25 - missing_penalty, 100))
            
            return {
                'phase': phase,
                'angle': angle,
                'confidence': confidence,
                'components': {
                    'trend': trend,
                    'seasonal': seasonal,
                    'residual': residual
                }
            }
        except Exception as e:
            st.warning(f"Error in decomposition: {str(e)}")
            return None
            
    except Exception as e:
        st.warning(f"Error analyzing market cycle: {str(e)}")
        return None

def calculate_market_imbalance(inventory_data, price_data, new_listings_data):
    """Calculate supply-demand imbalance metrics"""
    try:
        if inventory_data.empty or price_data.empty or new_listings_data.empty:
            return None
            
        # Calculate recent changes
        inv_change = inventory_data['Value'].pct_change(12).iloc[-1]
        price_change = price_data['Value'].pct_change(12).iloc[-1]
        new_listings_change = new_listings_data['Value'].pct_change(12).iloc[-1]
        
        # Calculate absorption rate (new listings vs inventory)
        absorption = new_listings_data['Value'].iloc[-1] / inventory_data['Value'].iloc[-1]
        
        # Calculate imbalance scores
        supply_score = inv_change * -100  # Negative inventory change = positive score
        demand_score = price_change * 100  # Positive price change = positive score
        
        # Combined market pressure score (-100 to 100)
        # Positive = seller's market, Negative = buyer's market
        market_pressure = (supply_score + demand_score) / 2
        
        return {
            'market_pressure': market_pressure,
            'absorption_rate': absorption,
            'inventory_change': inv_change,
            'price_change': price_change,
            'new_listings_change': new_listings_change
        }
    except Exception as e:
        st.warning(f"Error calculating market imbalance: {str(e)}")
        return None

def calculate_volatility_metrics(price_data, sold_above_data, periods=12):
    """Calculate market stability and risk metrics"""
    try:
        if price_data.empty:
            return None
            
        # Calculate price volatility
        returns = price_data['Value'].pct_change()
        volatility = returns.std() * np.sqrt(12)  # Annualized
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(periods).std() * np.sqrt(12)
        
        # Calculate price momentum
        momentum = returns.rolling(periods).mean()
        
        # Calculate market consistency
        if not sold_above_data.empty:
            sold_above_volatility = sold_above_data['Value'].rolling(periods).std()
        else:
            sold_above_volatility = None
        
        # Calculate risk score (0-100)
        # Lower volatility = better score
        risk_score = max(0, min(100 - volatility * 100, 100))
        
        return {
            'volatility': volatility,
            'rolling_volatility': rolling_vol,
            'momentum': momentum,
            'sold_above_volatility': sold_above_volatility,
            'risk_score': risk_score
        }
    except Exception as e:
        st.warning(f"Error calculating volatility metrics: {str(e)}")
        return None

def calculate_relative_metrics(metro_data, state_data, national_data):
    """Calculate relative performance metrics"""
    try:
        if metro_data.empty or state_data.empty or national_data.empty:
            return None
            
        # Calculate YoY changes
        metro_change = metro_data['Value'].pct_change(12).iloc[-1]
        state_change = state_data['Value'].pct_change(12).iloc[-1]
        national_change = national_data['Value'].pct_change(12).iloc[-1]
        
        # Calculate relative performance
        vs_state = metro_change - state_change
        vs_national = metro_change - national_change
        
        # Calculate percentile rank
        all_changes = pd.concat([
            metro_data['Value'].pct_change(12),
            state_data['Value'].pct_change(12),
            national_data['Value'].pct_change(12)
        ])
        percentile = stats.percentileofscore(all_changes.dropna(), metro_change)
        
        return {
            'metro_change': metro_change,
            'vs_state': vs_state,
            'vs_national': vs_national,
            'percentile': percentile
        }
    except Exception as e:
        st.warning(f"Error calculating relative metrics: {str(e)}")
        return None

def calculate_all_metro_scores():
    """Calculate investment scores for all metro areas"""
    try:
        # Get all metro areas
        metros = data['home_values']['RegionName'].unique()
        scores = []
        
        # Create a progress bar
        progress_bar = st.progress(0)
        total_metros = len(metros)
        
        for i, metro in enumerate(metros):
            try:
                # Get state for this metro
                state = data['home_values'][data['home_values']['RegionName'] == metro]['StateName'].iloc[0]
                
                # Clean up state name if needed
                state = re.sub(r'([A-Z]{2}),\s*\1', r'\1', state).strip()
                
                # Get metro data
                metro_mask = (data['home_values']['RegionName'] == metro)
                metro_prices = melt_data(data['home_values'][metro_mask])
                metro_rentals = melt_data(data['rentals'][metro_mask])
                
                # Calculate metrics
                pr_data = calculate_price_rent_ratio(metro_prices, metro_rentals)
                
                if pr_data:
                    # Get coordinates
                    coords = get_metro_coordinates(metro, state)
                    if coords:
                        lat, lon = coords
                        
                        scores.append({
                            'metro': metro,
                            'state': state,
                            'lat': lat,
                            'lon': lon,
                            'investment_score': pr_data['investment_score'],
                            'price_trend': pr_data['price_trend'],
                            'rent_trend': pr_data['rent_trend'],
                            'pr_ratio': pr_data['current_ratio']
                        })
                        
            except Exception as e:
                st.warning(f"Error processing {metro}: {str(e)}")
                continue
                
            # Update progress
            progress_bar.progress((i + 1) / total_metros)
        
        # Clear progress bar
        progress_bar.empty()
        
        if scores:
            return pd.DataFrame(scores)
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error calculating metro scores: {str(e)}")
        return pd.DataFrame()

def plot_market_pressure_gauge(imbalance_data):
    """Create a gauge chart for market pressure"""
    if not imbalance_data:
        return None
        
    pressure = imbalance_data['market_pressure']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pressure,
        title = {'text': "Market Pressure<br><sub>(-100 = Buyer's Market, +100 = Seller's Market)</sub>"},
        gauge = {
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-100, -50], 'color': 'lightgreen'},
                {'range': [-50, 0], 'color': 'paleturquoise'},
                {'range': [0, 50], 'color': 'lightyellow'},
                {'range': [50, 100], 'color': 'lightcoral'}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def plot_investment_score_gauge(pr_data):
    """Create a gauge chart for investment score"""
    if not pr_data:
        return None
        
    score = pr_data['investment_score']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': f"Investment Score<br><sub>Current P/R: {pr_data['current_ratio']:.1f}</sub>"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': 'lightcoral'},
                {'range': [33, 66], 'color': 'lightyellow'},
                {'range': [66, 100], 'color': 'lightgreen'}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def plot_market_cycle_gauge(cycle_data):
    """Create a gauge chart for market cycle position"""
    if not cycle_data:
        return None
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = cycle_data['angle'],
        title = {'text': f"Market Phase: {cycle_data['phase']}<br><sub>Confidence: {cycle_data['confidence']:.0f}%</sub>"},
        gauge = {
            'axis': {'range': [0, 360], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 90], 'color': 'lightgreen'},
                {'range': [90, 180], 'color': 'yellow'},
                {'range': [180, 270], 'color': 'orange'},
                {'range': [270, 360], 'color': 'lightblue'}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def plot_risk_gauge(volatility_data):
    """Create a gauge chart for risk score"""
    if not volatility_data:
        return None
        
    score = volatility_data['risk_score']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': f"Market Stability Score<br><sub>Volatility: {volatility_data['volatility']*100:.1f}%</sub>"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': 'lightcoral'},
                {'range': [33, 66], 'color': 'lightyellow'},
                {'range': [66, 100], 'color': 'lightgreen'}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def plot_relative_performance(relative_data):
    """Create a bullet chart for relative performance"""
    if not relative_data:
        return None
        
    fig = go.Figure(go.Indicator(
        mode = "number+gauge+delta",
        value = relative_data['percentile'],
        delta = {'reference': 50},
        title = {'text': f"Market Performance Percentile<br><sub>vs State: {relative_data['vs_state']*100:.1f}% | vs National: {relative_data['vs_national']*100:.1f}%</sub>"},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [0, 100]},
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': 50
            },
            'steps': [
                {'range': [0, 20], 'color': 'lightcoral'},
                {'range': [20, 40], 'color': 'lightyellow'},
                {'range': [40, 60], 'color': 'lightgreen'},
                {'range': [60, 80], 'color': 'lightyellow'},
                {'range': [80, 100], 'color': 'lightcoral'}
            ],
            'bar': {'color': "darkblue"}
        }
    ))
    
    fig.update_layout(height=200)
    return fig

# Sidebar for navigation
category = st.sidebar.selectbox(
    "Choose a Dashboard",
    ["Market Overview", "Market Analysis", "Market Activity", "Market Heatmap"]
)

# Main dashboard logic
if data_loaded:
    if category == "Market Heatmap":
        st.write("# 🗺️ National Market Investment Heatmap")
        
        # Calculate scores for all metros
        with st.spinner("Calculating investment scores for all metro areas..."):
            metro_scores_df = calculate_all_metro_scores()
            
        if not metro_scores_df.empty:
            # Debug information
            st.write(f"Found {len(metro_scores_df)} metros with valid scores")
            
            # Create the heatmap
            fig = px.scatter_mapbox(
                metro_scores_df,
                lat='lat',
                lon='lon',
                color='investment_score',
                size='investment_score',
                hover_name='metro',
                hover_data={
                    'lat': False,
                    'lon': False,
                    'investment_score': ':.1f',
                    'price_trend': ':.1f',
                    'rent_trend': ':.1f',
                    'pr_ratio': ':.1f',
                    'state': True
                },
                color_continuous_scale='RdYlGn',
                size_max=30,
                zoom=3,
                title='Real Estate Investment Opportunities'
            )
            
            fig.update_layout(
                mapbox_style="light",
                margin={"r":0,"t":0,"l":0,"b":0},
                height=800
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a data table below the map
            st.write("## Metro Area Investment Scores")
            st.dataframe(
                metro_scores_df.sort_values('investment_score', ascending=False),
                hide_index=True,
                column_config={
                    'metro': 'Metro Area',
                    'state': 'State',
                    'investment_score': st.column_config.NumberColumn(
                        'Investment Score',
                        help='0-100 score based on price-to-rent ratio and trends',
                        format="%.1f"
                    ),
                    'price_trend': st.column_config.NumberColumn(
                        'Price Trend (%)',
                        help='Year-over-year price change',
                        format="%.1f%%"
                    ),
                    'rent_trend': st.column_config.NumberColumn(
                        'Rent Trend (%)',
                        help='Year-over-year rent change',
                        format="%.1f%%"
                    ),
                    'pr_ratio': st.column_config.NumberColumn(
                        'P/R Ratio',
                        help='Price-to-annual-rent ratio',
                        format="%.1f"
                    ),
                    'lat': None,
                    'lon': None
                }
            )
        else:
            st.error("No metro areas found with valid investment scores. Please check the data.")
            
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
            st.write(f"# 📊 Market Overview: {selected_metro}")
            
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
                
    elif category == "Market Activity":
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
