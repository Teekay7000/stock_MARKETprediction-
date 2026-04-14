import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Load data
df = pd.read_csv('random_stock_market_dataset.csv')

# Data cleaning
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop_duplicates()

# Feature engineering
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

# Prepare features
feature_cols = ['Open', 'High', 'Low', 'Volume', 'Daily_Range', 'Price_Change',
                'Volume_MA_7', 'Close_MA_7', 'Volatility', 'Prev_Close', 
                'Prev_Volume', 'Day', 'Month', 'DayOfWeek']

X = df[feature_cols]
y = df['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
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
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# Model comparison
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'RMSE': [results[m]['RMSE'] for m in results.keys()],
    'MAE': [results[m]['MAE'] for m in results.keys()],
    'R2_Score': [results[m]['R2'] for m in results.keys()]
}).sort_values('R2_Score', ascending=False)

print("Model Performance Comparison:")
print(comparison_df)

best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

print(f"\nBest Model: {best_model_name}")
print(f"R² Score: {results[best_model_name]['R2']:.4f}")
print(f"RMSE: {results[best_model_name]['RMSE']:.2f}")
print(f"MAE: {results[best_model_name]['MAE']:.2f}")

# Feature importance (for tree-based models)
if 'Forest' in best_model_name or 'Boosting' in best_model_name:
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
    plt.title(f'Top 10 Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, results[best_model_name]['predictions'], alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price')
axes[0, 0].set_ylabel('Predicted Price')
axes[0, 0].set_title('Actual vs Predicted Prices')

# 2. Residual plot
residuals = y_test.values - results[best_model_name]['predictions']
axes[0, 1].scatter(results[best_model_name]['predictions'], residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted Price')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')

# 3. Close price over time
axes[0, 2].plot(df['Date'], df['Close'], linewidth=1)
axes[0, 2].set_xlabel('Date')
axes[0, 2].set_ylabel('Close Price')
axes[0, 2].set_title('Stock Price Trend')
axes[0, 2].tick_params(axis='x', rotation=45)

# 4. Distribution of Close prices
axes[1, 0].hist(df['Close'], bins=30, edgecolor='black')
axes[1, 0].set_xlabel('Close Price')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Closing Prices')

# 5. Correlation heatmap
corr_features = ['Open', 'High', 'Low', 'Close', 'Volume']
corr_matrix = df[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Feature Correlation Matrix')

# 6. Model comparison
model_names = list(results.keys())
r2_scores = [results[m]['R2'] for m in model_names]
axes[1, 2].barh(model_names, r2_scores)
axes[1, 2].set_xlabel('R² Score')
axes[1, 2].set_title('Model Performance Comparison')
axes[1, 2].set_xlim([0, 1])

plt.tight_layout()
plt.savefig('stock_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

# Save predictions
predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': results[best_model_name]['predictions'],
    'Error': y_test.values - results[best_model_name]['predictions']
})
predictions_df.to_csv('predictions.csv', index=False)

# Statistical summary
print("\n" + "="*50)
print("Statistical Summary:")
print("="*50)
print(f"Dataset size: {len(df)} records")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Average Close Price: ${df['Close'].mean():.2f}")
print(f"Price Volatility (std): ${df['Close'].std():.2f}")
print(f"Min Price: ${df['Close'].min():.2f}")
print(f"Max Price: ${df['Close'].max():.2f}")
print(f"Average Daily Range: ${df['Daily_Range'].mean():.2f}")
print(f"Average Volume: {df['Volume'].mean():,.0f}")

print("\n" + "="*50)
print("Files saved:")
print("="*50)
print("feature_importance.png")
print("stock_analysis_dashboard.png")
print("predictions.csv")