from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- STEP 1: DEFINE FEATURES & TARGET ---
X = df.drop(['project_id', 'total_project_effort', 'start_date'], axis=1)
y = df['total_project_effort']

# --- STEP 2: PREPROCESSING PIPELINE (Addressing RQ2) ---
# We treat numerical and categorical data differently as per your proposal [cite: 96, 114]

numeric_features = ['surface_area_m2', 'num_levels', 'floor_area_ratio']
categorical_features = ['project_type', 'material_type']

# Create transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()) # Z-score standardization [cite: 102]
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-Hot Encoding 
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 3: MODEL DEFINITIONS (Addressing RQ3) ---

# Model A: Linear Regression (Baseline)
lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Model B: Random Forest (Ensemble)
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

print("Phase 2 Complete: Pipelines constructed.")