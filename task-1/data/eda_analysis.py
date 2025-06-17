"""
Exploratory Data Analysis for ACIS Insurance Analytics
This script performs comprehensive EDA on insurance data to understand risk patterns and profitability drivers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InsuranceDataAnalyzer:
    def __init__(self, data_path: str):
        """Initialize the analyzer with data path."""
        self.data_path = Path(data_path)
        self.data = None
        self.load_data()
    
    def load_data(self) -> None:
        """Load and perform initial data preprocessing."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            self.data['TransactionMonth'] = pd.to_datetime(self.data['TransactionMonth'])
            logger.info(f"Successfully loaded {len(self.data)} records")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def check_data_quality(self) -> Dict:
        """Assess data quality including missing values and data types."""
        quality_report = {
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict(),
            'unique_values': {col: self.data[col].nunique() for col in self.data.columns}
        }
        return quality_report
    
    def calculate_loss_ratio(self, group_by: List[str] = None) -> pd.DataFrame:
        """Calculate loss ratio (TotalClaims / TotalPremium) by specified dimensions."""
        self.data['LossRatio'] = self.data['TotalClaims'] / self.data['TotalPremium']
        
        if group_by:
            return self.data.groupby(group_by)['LossRatio'].agg(['mean', 'std', 'count'])
        return self.data['LossRatio'].agg(['mean', 'std', 'count'])
    
    def analyze_financial_distributions(self) -> Dict[str, Tuple[float, float]]:
        """Analyze distributions of key financial variables."""
        financial_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
        stats = {}
        
        for col in financial_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            
            stats[col] = {
                'mean': self.data[col].mean(),
                'median': self.data[col].median(),
                'std': self.data[col].std(),
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers) / len(self.data)) * 100
            }
        
        return stats
    
    def analyze_temporal_trends(self) -> pd.DataFrame:
        """Analyze temporal trends in claims and premiums."""
        monthly_stats = self.data.groupby('TransactionMonth').agg({
            'TotalClaims': ['sum', 'mean', 'count'],
            'TotalPremium': ['sum', 'mean'],
            'LossRatio': 'mean'
        })
        
        # Calculate month-over-month changes
        for col in ['TotalClaims', 'TotalPremium']:
            monthly_stats[f'{col}_MoM_Change'] = monthly_stats[col]['sum'].pct_change()
        
        return monthly_stats
    
    def analyze_vehicle_risk(self) -> pd.DataFrame:
        """Analyze risk patterns by vehicle characteristics."""
        vehicle_risk = self.data.groupby(['VehicleType', 'VehicleMake']).agg({
            'TotalClaims': ['mean', 'sum', 'count'],
            'TotalPremium': ['mean', 'sum'],
            'LossRatio': 'mean'
        }).sort_values(('TotalClaims', 'sum'), ascending=False)
        
        return vehicle_risk
    
    def create_visualizations(self, output_dir: str = 'reports/figures/') -> None:
        """Create and save key visualizations."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Loss Ratio Distribution by Province
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.data, x='Province', y='LossRatio')
        plt.title('Loss Ratio Distribution by Province')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}loss_ratio_by_province.png')
        plt.close()
        
        # 2. Claims vs Premium Scatter Plot
        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            data=self.data,
            x='TotalPremium',
            y='TotalClaims',
            hue='VehicleType',
            alpha=0.6
        )
        plt.title('Claims vs Premium by Vehicle Type')
        plt.tight_layout()
        plt.savefig(f'{output_dir}claims_vs_premium.png')
        plt.close()
        
        # 3. Temporal Trend Analysis
        monthly_data = self.analyze_temporal_trends()
        plt.figure(figsize=(15, 7))
        plt.plot(monthly_data.index, monthly_data['LossRatio']['mean'], marker='o')
        plt.title('Monthly Loss Ratio Trend')
        plt.xlabel('Month')
        plt.ylabel('Loss Ratio')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}monthly_loss_ratio.png')
        plt.close()

def main():
    """Main execution function."""
    try:
        # Initialize analyzer
        analyzer = InsuranceDataAnalyzer('data/insurance_data.csv')
        
        # Perform analysis
        logger.info("Checking data quality...")
        quality_report = analyzer.check_data_quality()
        
        logger.info("Calculating loss ratios...")
        loss_ratios = analyzer.calculate_loss_ratio(['Province', 'VehicleType', 'Gender'])
        
        logger.info("Analyzing financial distributions...")
        financial_stats = analyzer.analyze_financial_distributions()
        
        logger.info("Analyzing temporal trends...")
        temporal_trends = analyzer.analyze_temporal_trends()
        
        logger.info("Analyzing vehicle risk patterns...")
        vehicle_risk = analyzer.analyze_vehicle_risk()
        
        logger.info("Creating visualizations...")
        analyzer.create_visualizations()
        
        # Save results
        output_dir = Path('reports/analysis/')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pd.DataFrame(quality_report).to_csv(output_dir / 'data_quality_report.csv')
        loss_ratios.to_csv(output_dir / 'loss_ratios_by_dimension.csv')
        pd.DataFrame(financial_stats).to_csv(output_dir / 'financial_statistics.csv')
        temporal_trends.to_csv(output_dir / 'temporal_trends.csv')
        vehicle_risk.to_csv(output_dir / 'vehicle_risk_analysis.csv')
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 