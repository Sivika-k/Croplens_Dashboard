# train_gbm.py - Script to train and save the Gradient Boosting Model

import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. CONFIGURATION ---
DATA_PATH = 'data/rice_data.csv'
MODEL_SAVE_PATH = 'models/gbm_rice_predictor.joblib'

print("--- Starting Gradient Boosting Model Training ---")

# --- 2. DATA LOADING & PREPARATION (Matching app.py logic) ---
try:
    df = pd.read_csv(DATA_PATH)
    # Rename columns to match the names used in app.py
    df.rename(columns={
        'Date': 'date',                           
        'Rice Variety': 'variety',                
        'District': 'district',                   
        'Market Name': 'market',                  
        'Rice Price (₹/Quintal)': 'modal_price'   
    }, inplace=True)
    df['date'] = pd.to_datetime(df['date'])

    # Feature Engineering (as defined in app.py's get_prediction function)
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day

    # Define Features (X) and Target (y)
    # We use modal_price as the target
    y = df['modal_price']
    
    # We drop the original columns that are not directly used in the model input (Year, Month, Day are used instead)
    X = df.drop(columns=['date', 'modal_price']) 

    print("Data loaded and features created successfully.")

except Exception as e:
    print(f"FATAL ERROR: Could not load data for training. Check {DATA_PATH}. Error: {e}")
    exit()

# --- 3. CATEGORICAL ENCODING (Crucial step for GBM) ---
# We use One-Hot Encoding as assumed in the app.py prediction logic
X_encoded = pd.get_dummies(X, columns=['variety', 'district', 'market'], dtype=int)


# --- 4. MODEL TRAINING ---

# Drop any non-numeric or extraneous columns before training
# (Example: We might drop the original variety/district/market if they somehow remain)
X_train = X_encoded.drop(columns=X_encoded.select_dtypes(include=['object']).columns, errors='ignore')

# Split data (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, random_state=42)

print(f"Starting GBM training with {len(X_train)} samples...")

# Initialize the Gradient Boosting Regressor (using reasonable parameters)
gbm = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train the model
gbm.fit(X_train, y_train)

# (Optional: Evaluate model performance)
# score = gbm.score(X_test, y_test)
# print(f"Model R-squared score on test data: {score:.4f}")

# --- 5. SAVE THE MODEL ---
try:
    # Save the trained model object using joblib
    joblib.dump(gbm, MODEL_SAVE_PATH)
    print(f"\n✅ SUCCESS! Trained Gradient Boosting Model saved to {MODEL_SAVE_PATH}")
    print("Now, restart your Flask server (python app.py) to load the model.")
except Exception as e:
    print(f"❌ ERROR: Could not save model. Check 'models/' folder permission. Error: {e}")