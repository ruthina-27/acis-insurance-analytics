import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class PremiumPredictor:
    def __init__(self):
        """
        Initialize the premium predictor with necessary preprocessing and model components.
        """
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for premium prediction.
        
        Args:
            data (pd.DataFrame): Preprocessed insurance data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variable
        """
        # Define feature groups
        numeric_features = [
            'SumInsured', 'CalculatedPremiumPerTerm', 'ExcessSelected',
            'RegistrationYear', 'Cylinders', 'Cubiccapacity', 'Kilowatts',
            'NumberOfDoors', 'CustomValueEstimate', 'CapitalOutstanding'
        ]
        
        categorical_features = [
            'Province', 'Gender', 'MaritalStatus', 'VehicleType',
            'Bodytype', 'CoverType', 'CoverCategory'
        ]
        
        # Select features
        X = data[numeric_features + categorical_features]
        y = data['TotalPremium']
        
        # Store feature names
        self.feature_names = numeric_features + categorical_features
        
        return X, y
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline for numeric and categorical features.
        
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        numeric_features = [
            'SumInsured', 'CalculatedPremiumPerTerm', 'ExcessSelected',
            'RegistrationYear', 'Cylinders', 'Cubiccapacity', 'Kilowatts',
            'NumberOfDoors', 'CustomValueEstimate', 'CapitalOutstanding'
        ]
        
        categorical_features = [
            'Province', 'Gender', 'MaritalStatus', 'VehicleType',
            'Bodytype', 'CoverType', 'CoverCategory'
        ]
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the premium prediction model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        # Create preprocessing pipeline
        self.preprocessor = self.create_preprocessing_pipeline()
        
        # Create model pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ))
        ])
        
        # Train model
        self.model.fit(X, y)
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the premium prediction model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Dict[str, float]: Model evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=5,
            scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        return {
            'rmse': rmse,
            'r2_score': r2,
            'cv_rmse': cv_rmse
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if self.model is None:
            raise Exception("Model not trained. Call train_model() first.")
        
        # Get feature names after preprocessing
        feature_names = (
            self.model.named_steps['preprocessor']
            .named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(self.feature_names)
        )
        
        # Get feature importance
        importance = self.model.named_steps['regressor'].feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict_premium(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict optimal premium values for new data.
        
        Args:
            X (pd.DataFrame): Feature matrix for prediction
            
        Returns:
            np.ndarray: Predicted premium values
        """
        if self.model is None:
            raise Exception("Model not trained. Call train_model() first.")
        
        return self.model.predict(X)
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise Exception("Model not trained. Call train_model() first.")
        
        joblib.dump(self.model, path)
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to the saved model
        """
        self.model = joblib.load(path) 