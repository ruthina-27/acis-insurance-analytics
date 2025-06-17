import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from pathlib import Path

class InsuranceDataProcessor:
    def __init__(self, data_path: str):
        """
        Initialize the data processor with the path to the insurance data.
        
        Args:
            data_path (str): Path to the insurance data file
        """
        self.data_path = Path(data_path)
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the insurance data from the specified path.
        
        Returns:
            pd.DataFrame: Loaded insurance data
        """
        try:
            self.data = pd.read_csv(self.data_path)
            return self.data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the insurance data by handling missing values and converting data types.
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.data is None:
            raise Exception("Data not loaded. Call load_data() first.")
            
        # Create a copy to avoid modifying the original data
        df = self.data.copy()
        
        # Convert date columns
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get the feature groups for different aspects of the insurance data.
        
        Returns:
            Dict[str, List[str]]: Dictionary of feature groups
        """
        return {
            'policy': ['UnderwrittenCoverID', 'PolicyID'],
            'client': [
                'IsVATRegistered', 'Citizenship', 'LegalType', 'Title',
                'Language', 'Bank', 'AccountType', 'MaritalStatus', 'Gender'
            ],
            'location': [
                'Country', 'Province', 'PostalCode',
                'MainCrestaZone', 'SubCrestaZone'
            ],
            'vehicle': [
                'ItemType', 'Mmcode', 'VehicleType', 'RegistrationYear',
                'Make', 'Model', 'Cylinders', 'Cubiccapacity', 'Kilowatts',
                'Bodytype', 'NumberOfDoors', 'VehicleIntroDate',
                'CustomValueEstimate', 'AlarmImmobiliser', 'TrackingDevice',
                'CapitalOutstanding', 'NewVehicle', 'WrittenOff', 'Rebuilt',
                'Converted', 'CrossBorder', 'NumberOfVehiclesInFleet'
            ],
            'plan': [
                'SumInsured', 'TermFrequency', 'CalculatedPremiumPerTerm',
                'ExcessSelected', 'CoverCategory', 'CoverType', 'CoverGroup',
                'Section', 'Product', 'StatutoryClass', 'StatutoryRiskType'
            ],
            'financial': ['TotalPremium', 'TotalClaims']
        }
    
    def calculate_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk metrics for the insurance data.
        
        Args:
            df (pd.DataFrame): Preprocessed insurance data
            
        Returns:
            pd.DataFrame: Data with additional risk metrics
        """
        # Calculate claim frequency
        df['ClaimFrequency'] = df['TotalClaims'] / df['TotalPremium']
        
        # Calculate risk score (custom metric)
        df['RiskScore'] = (df['TotalClaims'] / df['SumInsured']) * 100
        
        # Calculate profit margin
        df['ProfitMargin'] = (df['TotalPremium'] - df['TotalClaims']) / df['TotalPremium']
        
        return df 