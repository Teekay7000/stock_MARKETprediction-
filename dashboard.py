import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Stock Market Analysis", layout="wide", page_icon="📈")

# Title
st.title("📈 Stock Market Data Science Dashboard")
st.markdown("---")

# Load data function
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv('random_stock_market_dataset.csv')
        except FileNotFoundError:
            return None
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

# Feature engineering function
@st.cache_data
def engineer_features(df):
    df = df.copy()
    df['Daily_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    df['Price_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    df['Volume_MA_7'] = df['Volume'].rolling(window=7, min_periods=1).mean()
    df['Close_MA_7'] = df['Close'].rolling(window=7, min_periods=1).mean()
    df['Volatility'] = df['Close'].rolling(window=7, min_periods=1).std()
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Volume'] = df['Volume'].shift(1)
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df = df.fillna(method='bfill')
    return df

# Train models function
@st.cache_data
def train_models(df):
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'Daily_Range', 'Price_Change',
                    'Volume_MA_7', 'Close_MA_7', 'Volatility', 'Prev_Close', 
                    'Prev_Volume', 'Day', 'Month', 'DayOfWeek']
    
    X = df[feature_cols]
    y = df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = {}
    
    for name, model in models.items():
        if name in ['Linear Regression', 'Ridge', 'Lasso']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'actual': y_test.values,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    return results, feature_cols, X_test

# Sidebar
st.sidebar.header("Dashboard Controls")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

# Load and prepare data
df = load_data(uploaded_file)

# Check if data is loaded
if df is None:
    st.warning("Please upload a CSV file or place 'random_stock_market_dataset.csv' in the same folder.")
    st.info("Your CSV should have columns: Date, Open, High, Low, Close, Volume")
    st.stop()

df_engineered = engineer_features(df)

page = st.sidebar.selectbox("Select Page", 
                             ["Overview", "Exploratory Analysis", "ML Models", "Predictions"])

# Overview Page
if page == "Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Avg Close Price", f"${df['Close'].mean():.2f}")
    with col3:
        st.metric("Max Price", f"${df['Close'].max():.2f}")
    with col4:
        st.metric("Min Price", f"${df['Close'].min():.2f}")
    
    st.subheader("Stock Price Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price',
                             line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=df_engineered['Date'], y=df_engineered['Close_MA_7'], 
                             mode='lines', name='7-Day MA',
                             line=dict(color='#ff7f0e', width=2, dash='dash')))
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Price ($)",
                      hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig = px.histogram(df, x='Close', nbins=30, title="Distribution of Closing Prices")
        fig.update_layout(height=350, xaxis_title="Close Price", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Volume Over Time")
        fig = px.bar(df, x='Date', y='Volume', title="Trading Volume")
        fig.update_layout(height=350, xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Data Sample")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# Exploratory Analysis Page
elif page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    st.subheader("Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Price ($)",
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Matrix")
        corr_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        corr_matrix = df[corr_features].corr()
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Feature Correlation")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Daily Price Change")
        fig = px.histogram(df_engineered, x='Price_Change_Pct', nbins=30,
                          title="Distribution of Daily % Change")
        fig.update_layout(height=400, xaxis_title="% Change", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Technical Indicators")
    
    selected_indicator = st.selectbox("Select Indicator", 
                                      ['Daily Range', 'Volatility', 'Volume MA 7'])
    
    if selected_indicator == 'Daily Range':
        fig = px.line(df_engineered, x='Date', y='Daily_Range', 
                      title='Daily Price Range Over Time')
    elif selected_indicator == 'Volatility':
        fig = px.line(df_engineered, x='Date', y='Volatility',
                      title='7-Day Volatility Over Time')
    else:
        fig = px.line(df_engineered, x='Date', y='Volume_MA_7',
                      title='7-Day Volume Moving Average')
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ML Models Page
elif page == "ML Models":
    st.header("Machine Learning Models")
    
    with st.spinner("Training models..."):
        results, feature_cols, X_test = train_models(df_engineered)
    
    st.success("Models trained successfully!")
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE': [results[m]['RMSE'] for m in results.keys()],
        'MAE': [results[m]['MAE'] for m in results.keys()],
        'R² Score': [results[m]['R2'] for m in results.keys()]
    }).sort_values('R² Score', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(comparison_df, x='Model', y='R² Score', 
                     title='Model Performance (R² Score)',
                     color='R² Score', color_continuous_scale='viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Best model details
    best_model_name = comparison_df.iloc[0]['Model']
    best_result = results[best_model_name]
    
    st.subheader(f"Best Model: {best_model_name}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{best_result['R2']:.4f}")
    with col2:
        st.metric("RMSE", f"${best_result['RMSE']:.2f}")
    with col3:
        st.metric("MAE", f"${best_result['MAE']:.2f}")
    
    # Feature importance
    if 'Forest' in best_model_name or 'Boosting' in best_model_name:
        st.subheader("Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_result['model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance.head(10), x='Importance', y='Feature',
                     orientation='h', title='Top 10 Important Features')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Actual vs Predicted
    st.subheader("Actual vs Predicted Prices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=best_result['actual'], y=best_result['predictions'],
                                mode='markers', name='Predictions',
                                marker=dict(size=8, opacity=0.6)))
        fig.add_trace(go.Scatter(x=[best_result['actual'].min(), best_result['actual'].max()],
                                y=[best_result['actual'].min(), best_result['actual'].max()],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(height=400, xaxis_title="Actual Price", yaxis_title="Predicted Price",
                         title="Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        residuals = best_result['actual'] - best_result['predictions']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=best_result['predictions'], y=residuals,
                                mode='markers', marker=dict(size=8, opacity=0.6)))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=400, xaxis_title="Predicted Price", yaxis_title="Residuals",
                         title="Residual Plot")
        st.plotly_chart(fig, use_container_width=True)

# Predictions Page
elif page == "Predictions":
    st.header("Price Predictions")
    
    with st.spinner("Training models..."):
        results, feature_cols, X_test = train_models(df_engineered)
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['R2'])
    best_result = results[best_model_name]
    
    st.info(f"Using best model: **{best_model_name}** (R² = {best_result['R2']:.4f})")
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'Actual Price': best_result['actual'],
        'Predicted Price': best_result['predictions'],
        'Error': best_result['actual'] - best_result['predictions'],
        'Error %': ((best_result['actual'] - best_result['predictions']) / best_result['actual'] * 100)
    })
    
    st.subheader("Prediction Results")
    st.dataframe(predictions_df.head(20), use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Error", f"${predictions_df['Error'].mean():.2f}")
    with col2:
        st.metric("Std Error", f"${predictions_df['Error'].std():.2f}")
    with col3:
        st.metric("Max Overestimate", f"${predictions_df['Error'].min():.2f}")
    with col4:
        st.metric("Max Underestimate", f"${predictions_df['Error'].max():.2f}")
    
    st.subheader("Prediction Timeline")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(best_result['actual']))), 
                            y=best_result['actual'],
                            mode='lines+markers', name='Actual Price',
                            line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=list(range(len(best_result['predictions']))), 
                            y=best_result['predictions'],
                            mode='lines+markers', name='Predicted Price',
                            line=dict(color='red', width=2, dash='dash')))
    fig.update_layout(height=450, xaxis_title="Test Sample Index", yaxis_title="Price ($)",
                     hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Download predictions
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="stock_predictions.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("**Stock Market Analysis Dashboard** | Built with Streamlit & Scikit-learn")