import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import folium
from folium.plugins import HeatMap

class InsuranceVisualizer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the visualizer with the preprocessed data.
        
        Args:
            data (pd.DataFrame): Preprocessed insurance data
        """
        self.data = data
        plt.style.use('seaborn')
        
    def plot_risk_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of risk scores.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='RiskScore', bins=50)
        plt.title('Distribution of Risk Scores')
        plt.xlabel('Risk Score')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_provincial_risk_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Create a box plot comparing risk scores across provinces.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.data, x='Province', y='RiskScore')
        plt.title('Risk Score Distribution by Province')
        plt.xticks(rotation=45)
        plt.xlabel('Province')
        plt.ylabel('Risk Score')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_gender_risk_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Create a box plot comparing risk scores between genders.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.data, x='Gender', y='RiskScore')
        plt.title('Risk Score Distribution by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Risk Score')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_premium_vs_claims(self, save_path: Optional[str] = None) -> None:
        """
        Create a scatter plot of premium vs claims.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.data,
            x='TotalPremium',
            y='TotalClaims',
            alpha=0.5
        )
        plt.title('Premium vs Claims')
        plt.xlabel('Total Premium')
        plt.ylabel('Total Claims')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 10,
                              save_path: Optional[str] = None) -> None:
        """
        Create a bar plot of feature importance.
        
        Args:
            importance_df (pd.DataFrame): Feature importance DataFrame
            top_n (int): Number of top features to display
            save_path (Optional[str]): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=importance_df.head(top_n),
            x='importance',
            y='feature'
        )
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def create_risk_heatmap(self, save_path: Optional[str] = None) -> None:
        """
        Create a heatmap of risk scores by province and vehicle type.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        # Calculate mean risk score by province and vehicle type
        risk_matrix = self.data.pivot_table(
            values='RiskScore',
            index='Province',
            columns='VehicleType',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            risk_matrix,
            annot=True,
            cmap='YlOrRd',
            fmt='.2f'
        )
        plt.title('Risk Score Heatmap by Province and Vehicle Type')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_zipcode_risk_map(self, save_path: Optional[str] = None) -> None:
        """
        Create a map visualization of risk scores by zipcode.
        
        Args:
            save_path (Optional[str]): Path to save the map
        """
        # Calculate mean risk score by zipcode
        zipcode_risk = self.data.groupby('PostalCode')['RiskScore'].mean().reset_index()
        
        # Create a map centered on South Africa
        m = folium.Map(location=[-30.5595, 22.9375], zoom_start=5)
        
        # Add heatmap layer
        heat_data = [[row['PostalCode'], row['RiskScore']] for _, row in zipcode_risk.iterrows()]
        HeatMap(heat_data).add_to(m)
        
        if save_path:
            m.save(save_path)
    
    def plot_time_series_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Create time series plots of premium and claims.
        
        Args:
            save_path (Optional[str]): Path to save the plot
        """
        # Calculate monthly aggregates
        monthly_data = self.data.groupby('TransactionMonth').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot premium
        sns.lineplot(data=monthly_data, x='TransactionMonth', y='TotalPremium', ax=ax1)
        ax1.set_title('Monthly Total Premium')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Total Premium')
        
        # Plot claims
        sns.lineplot(data=monthly_data, x='TransactionMonth', y='TotalClaims', ax=ax2)
        ax2.set_title('Monthly Total Claims')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Total Claims')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close() 