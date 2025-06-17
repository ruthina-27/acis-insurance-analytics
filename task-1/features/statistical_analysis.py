import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

class StatisticalAnalyzer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the statistical analyzer with the preprocessed data.
        
        Args:
            data (pd.DataFrame): Preprocessed insurance data
        """
        self.data = data
        
    def test_provincial_risk_differences(self) -> Dict[str, float]:
        """
        Test for risk differences across provinces using ANOVA.
        
        Returns:
            Dict[str, float]: Test results including F-statistic and p-value
        """
        # Group data by province and calculate mean risk scores
        province_groups = [group for _, group in self.data.groupby('Province')]
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*[group['RiskScore'] for group in province_groups])
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def test_zipcode_risk_differences(self) -> Dict[str, float]:
        """
        Test for risk differences between zipcodes using ANOVA.
        
        Returns:
            Dict[str, float]: Test results including F-statistic and p-value
        """
        # Group data by zipcode and calculate mean risk scores
        zipcode_groups = [group for _, group in self.data.groupby('PostalCode')]
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*[group['RiskScore'] for group in zipcode_groups])
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def test_zipcode_profit_differences(self) -> Dict[str, float]:
        """
        Test for profit margin differences between zipcodes using ANOVA.
        
        Returns:
            Dict[str, float]: Test results including F-statistic and p-value
        """
        # Group data by zipcode and calculate mean profit margins
        zipcode_groups = [group for _, group in self.data.groupby('PostalCode')]
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*[group['ProfitMargin'] for group in zipcode_groups])
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def test_gender_risk_differences(self) -> Dict[str, float]:
        """
        Test for risk differences between genders using t-test.
        
        Returns:
            Dict[str, float]: Test results including t-statistic and p-value
        """
        # Split data by gender
        male_data = self.data[self.data['Gender'] == 'Male']['RiskScore']
        female_data = self.data[self.data['Gender'] == 'Female']['RiskScore']
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(male_data, female_data)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def fit_zipcode_regression_models(self) -> Dict[str, Dict[str, float]]:
        """
        Fit linear regression models for total claims prediction by zipcode.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of regression results by zipcode
        """
        results = {}
        
        for zipcode, group in self.data.groupby('PostalCode'):
            # Prepare features for regression
            X = group[['SumInsured', 'CalculatedPremiumPerTerm', 'ExcessSelected']]
            y = group['TotalClaims']
            
            # Add constant for intercept
            X = sm.add_constant(X)
            
            # Fit regression model
            model = sm.OLS(y, X).fit()
            
            results[zipcode] = {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'coefficients': model.params.to_dict()
            }
        
        return results
    
    def calculate_feature_importance(self, target: str = 'TotalClaims') -> pd.DataFrame:
        """
        Calculate feature importance using correlation analysis.
        
        Args:
            target (str): Target variable for importance calculation
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        # Select numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Calculate correlations with target
        correlations = self.data[numeric_cols].corr()[target].abs()
        
        # Sort by absolute correlation
        importance = correlations.sort_values(ascending=False)
        
        return pd.DataFrame({
            'feature': importance.index,
            'importance': importance.values
        }) 