import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import json
import zipfile
import os

# 1. Load Data
train_path = 'dataset/train.csv'
test_path = 'dataset/test.csv'
sample_sub_path = 'dataset/sample_submission.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 2. Preprocessing & Feature Engineering
def preprocess(df):
    df = df.copy()
    
    # Time features
    if 'timestamp' in df.columns:
        # e.g., '0:0' -> hour=0, minute=0
        time_split = df['timestamp'].str.split(':', expand=True)
        df['hour'] = time_split[0].astype(float)
        df['minute'] = time_split[1].astype(float)
        df = df.drop(columns=['timestamp'])
    else:
        df['hour'] = 0
        df['minute'] = 0
        
    # Drop geohash for simplicity (high cardinality, or we could frequency encode it)
    # Let's use frequency encoding for geohash
    if 'geohash' in df.columns:
        freq = df['geohash'].value_counts()
        df['geohash_freq'] = df['geohash'].map(freq)
        df = df.drop(columns=['geohash'])
        
    # Convert categorical to category dtype for HistGradientBoostingRegressor
    cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].astype('category')
            
    # Set index
    if 'Index' in df.columns:
        df = df.set_index('Index')
        
    return df

X_train_full = preprocess(train_df.drop(columns=['demand']))
y_train_full = train_df['demand']
X_test = preprocess(test_df)

# 3. Validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Get categorical feature indices for HistGradientBoostingRegressor
cat_features = [X_tr.columns.get_loc(col) for col in ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather'] if col in X_tr.columns]

model = HistGradientBoostingRegressor(categorical_features=cat_features, random_state=42, max_iter=200)
model.fit(X_tr, y_tr)

val_preds = model.predict(X_val)
r2 = r2_score(y_val, val_preds)
score = max(0, 100 * r2)
print(f"Validation R2 Score: {r2:.4f}")
print(f"Hackathon Evaluation Score: {score:.4f}")

# 4. Train on full dataset and predict
print("Training on full dataset...")
model.fit(X_train_full, y_train_full)
test_preds = model.predict(X_test)

# 5. Create Submission File
sub_df = pd.DataFrame({
    'Index': test_df['Index'],
    'demand': test_preds
})
sub_df.to_csv('predictions.csv', index=False)
print("predictions.csv created successfully.")

# 6. Create Approach Document
approach_text = """
Traffic Demand Prediction Approach

1. Feature Engineering:
- Timestamp Parsing: Extracted 'hour' and 'minute' from the 'timestamp' column.
- Categorical Encoding: Used categorical types for 'RoadType', 'LargeVehicles', 'Landmarks', and 'Weather' to be natively handled by the HistGradientBoostingRegressor.
- High Cardinality: Frequency encoded 'geohash' to capture location popularity while keeping the feature space small.
- Handled Missing Values: Left them as NaNs, which are natively supported by HistGradientBoostingRegressor.

2. Modeling:
- Model: HistGradientBoostingRegressor from scikit-learn.
- Why: It handles missing values and categorical features natively without explicit imputation or one-hot encoding, making it robust and fast for tabular data.
- Hyperparameters: default with max_iter=200 and random_state=42 for reproducibility.

3. Tools Used:
- Python
- pandas for data manipulation
- scikit-learn for model training and evaluation
"""

with open('approach.txt', 'w') as f:
    f.write(approach_text.strip())

# 7. Create Jupyter Notebook Source Code representation
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Traffic Demand Prediction\n",
    "This notebook contains the source code for the traffic demand prediction task."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import HistGradientBoostingRegressor\n",
    "from sklearn.metrics import r2_score\n",
    "\n",
    "train_df = pd.read_csv('train.csv')\n",
    "test_df = pd.read_csv('test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess(df):\n",
    "    df = df.copy()\n",
    "    if 'timestamp' in df.columns:\n",
    "        time_split = df['timestamp'].str.split(':', expand=True)\n",
    "        df['hour'] = time_split[0].astype(float)\n",
    "        df['minute'] = time_split[1].astype(float)\n",
    "        df = df.drop(columns=['timestamp'])\n",
    "    else:\n",
    "        df['hour'], df['minute'] = 0, 0\n",
    "    if 'geohash' in df.columns:\n",
    "        freq = df['geohash'].value_counts()\n",
    "        df['geohash_freq'] = df['geohash'].map(freq)\n",
    "        df = df.drop(columns=['geohash'])\n",
    "    for col in ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']:\n",
    "        if col in df.columns:\n",
    "            df[col] = df[col].astype(str).astype('category')\n",
    "    if 'Index' in df.columns:\n",
    "        df = df.set_index('Index')\n",
    "    return df\n",
    "\n",
    "X_train = preprocess(train_df.drop(columns=['demand']))\n",
    "y_train = train_df['demand']\n",
    "X_test = preprocess(test_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "cat_features = [X_train.columns.get_loc(col) for col in ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather'] if col in X_train.columns]\n",
    "model = HistGradientBoostingRegressor(categorical_features=cat_features, random_state=42, max_iter=200)\n",
    "model.fit(X_train, y_train)\n",
    "test_preds = model.predict(X_test)\n",
    "\n",
    "sub_df = pd.DataFrame({'Index': test_df['Index'], 'demand': test_preds})\n",
    "sub_df.to_csv('predictions.csv', index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('solution.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

# 8. Create Zip Archive
with zipfile.ZipFile('submission.zip', 'w') as zipf:
    zipf.write('predictions.csv')
    zipf.write('approach.txt')
    zipf.write('solution.ipynb')

print("submission.zip created successfully.")
