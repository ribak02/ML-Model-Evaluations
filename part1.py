import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import  KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder


# Custom transformer for datetime features
class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert the input to a datetime series
        if isinstance(X, pd.DataFrame):
            X = pd.to_datetime(X.iloc[:, 0])  # Use iloc to ensure a Series is returned if X is a DataFrame
        else:  # assuming X is a Series if not a DataFrame
            X = pd.to_datetime(X)
        return X.dt.year.to_frame()  # Convert to DataFrame with one column for compatibility with ColumnTransformer

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """A custom transformer to select subsets of data."""
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names]

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """A custom transformer for explicit data preparation."""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Explicitly handle missing values and type conversion
        X = X.copy()  # Create a copy to avoid SettingWithCopyWarning
        for column in X.columns:
            # Ensure all columns are of type object
            X[column] = X[column].astype('object')
            # Replace NaNs with a placeholder
            X[column] = X[column].fillna('missing')
            # Optionally, use infer_objects() to infer better dtypes for object columns
            X[column] = X[column].infer_objects()
        return X
    
# Load hyperparameters from file
def load_hyperparams(model_type):
    try:
        with open(f'{model_type}_best_hyperparams.json', 'r') as f:
            hyperparams = json.load(f)
        return hyperparams
    except FileNotFoundError:
        print(f"No hyperparameter file found for {model_type}. Using default parameters.")
        return {}

# Load datasets
def load_data(train_input_path, train_labels_path, test_input_path):
    train_df = pd.read_csv(train_input_path)
    train_labels = pd.read_csv(train_labels_path)
    test_df = pd.read_csv(test_input_path)
    return train_df, train_labels, test_df

# Preprocess function
def preprocess_data(train_df, test_df, num_preprocessing, cat_preprocessing):
    numeric_features = train_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = train_df.select_dtypes(include=['object']).columns
    datetime_features = ['date_recorded']

    # Numeric preprocessing
    if num_preprocessing == 'StandardScaler':
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
    else:  # No scaling
        numeric_transformer = SimpleImputer(strategy='median')

    # Categorical preprocessing
    if cat_preprocessing == 'OneHotEncoder':
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
    elif cat_preprocessing == 'OrdinalEncoder':
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder())
        ])
    elif cat_preprocessing == 'TargetEncoder':
        categorical_transformer = Pipeline(steps=[
            ('selector', DataFrameSelector(categorical_features)),  # Select only categorical columns
            ('preprocessor', CustomPreprocessor()),  # Explicitly prepare data
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('target', TargetEncoder(handle_unknown='ignore', handle_missing='ignore')),
            ('fillna', SimpleImputer(strategy='constant', fill_value=0))  # Filling any remaining NaNs after encoding
        ])
    else: # No encoding
        categorical_transformer = SimpleImputer(strategy='most_frequent')

    # Datetime preprocessing
    datetime_transformer = Pipeline(steps=[
        ('date', DateTransformer()),
        ('imputer', SimpleImputer(strategy='most_frequent'))])

    # Column transformer to apply transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('date', datetime_transformer, datetime_features)])

    return preprocessor

# Model selection based on input argument
def select_model(model_type):
    # Load hyperparameters from file
    hyperparams = load_hyperparams(model_type)

    # Define model-specific hyperparameters
    if model_type == 'RandomForestClassifier':
        model_hyperparams = {key: hyperparams[key] for key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'] if key in hyperparams}
        model = RandomForestClassifier(**model_hyperparams, random_state=42)
    elif model_type == 'LogisticRegression':
        model_hyperparams = {key: hyperparams[key] for key in ['C', 'solver', 'penalty'] if key in hyperparams}
        model = LogisticRegression(**model_hyperparams, solver='liblinear', max_iter=1000, random_state=42)
    elif model_type == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == 'HistGradientBoostingClassifier':
        model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    elif model_type == 'MLPClassifier':
        model = MLPClassifier(max_iter=500, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model

def main(train_input_path, train_labels_path, test_input_path, numerical_preprocessing, categorical_preprocessing, model_type, test_prediction_output_file):
    # Load datasets
    train_df, train_labels, test_df = load_data(train_input_path, train_labels_path, test_input_path)

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the target variable
    y_train = label_encoder.fit_transform(train_labels['status_group'])

    # Preprocess data
    preprocessor = preprocess_data(train_df, test_df, numerical_preprocessing, categorical_preprocessing)

    model = select_model(model_type)
    
    # Define full pipeline (preprocessing + model)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    # Cross-validation using the features and y_train
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, train_df, y_train, cv=kf, scoring='accuracy', verbose=1, n_jobs=-1)
    
    # Output the cross-validation scores
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores)}")
    
    # Train the model using the full training set
    pipeline.fit(train_df, y_train)
    
    # Generating predictions for the test set, ensure test_df is only features
    test_predictions = pipeline.predict(test_df)

    # Convert numerical predictions back to original string labels
    test_predictions_labels = label_encoder.inverse_transform(test_predictions)

    # Combine the 'id' column from the test set with the predictions
    output = pd.DataFrame({
    'id': test_df['id'],
    'status_group': test_predictions_labels
    })
    
    # Output predictions
    output.to_csv(test_prediction_output_file, index=False)
    print(f"Predictions written to {test_prediction_output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 8:
        print("Usage: python part1.py <train-input-file> <train-labels-file> <test-input-file> <numerical-preprocessing> <categorical-preprocessing> <model-type> <test-prediction-output-file>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])