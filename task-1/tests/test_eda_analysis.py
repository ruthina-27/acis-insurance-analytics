"""
Tests for the EDA analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.eda_analysis import InsuranceDataAnalyzer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = pd.DataFrame({
        'TransactionMonth': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'Province': np.random.choice(['A', 'B', 'C'], 12),
        'VehicleType': np.random.choice(['Sedan', 'SUV', 'Truck'], 12),
        'VehicleMake': np.random.choice(['Toyota', 'Honda', 'Ford'], 12),
        'Gender': np.random.choice(['M', 'F'], 12),
        'TotalPremium': np.random.uniform(1000, 5000, 12),
        'TotalClaims': np.random.uniform(0, 3000, 12),
        'CustomValueEstimate': np.random.uniform(10000, 50000, 12)
    })
    return data

@pytest.fixture
def temp_csv(tmp_path, sample_data):
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "test_insurance_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path

def test_load_data(temp_csv):
    """Test data loading functionality."""
    analyzer = InsuranceDataAnalyzer(temp_csv)
    assert len(analyzer.data) == 12
    assert 'TransactionMonth' in analyzer.data.columns
    assert pd.api.types.is_datetime64_any_dtype(analyzer.data['TransactionMonth'])

def test_check_data_quality(temp_csv):
    """Test data quality assessment."""
    analyzer = InsuranceDataAnalyzer(temp_csv)
    quality_report = analyzer.check_data_quality()
    
    assert 'missing_values' in quality_report
    assert 'data_types' in quality_report
    assert 'unique_values' in quality_report
    assert all(count == 0 for count in quality_report['missing_values'].values())

def test_calculate_loss_ratio(temp_csv):
    """Test loss ratio calculation."""
    analyzer = InsuranceDataAnalyzer(temp_csv)
    
    # Test overall loss ratio
    overall_ratio = analyzer.calculate_loss_ratio()
    assert isinstance(overall_ratio, pd.Series)
    assert 'mean' in overall_ratio.index
    
    # Test grouped loss ratio
    grouped_ratio = analyzer.calculate_loss_ratio(['Province'])
    assert isinstance(grouped_ratio, pd.DataFrame)
    assert all(col in grouped_ratio.columns for col in ['mean', 'std', 'count'])

def test_analyze_financial_distributions(temp_csv):
    """Test financial distribution analysis."""
    analyzer = InsuranceDataAnalyzer(temp_csv)
    stats = analyzer.analyze_financial_distributions()
    
    expected_metrics = ['mean', 'median', 'std', 'outliers_count', 'outliers_percentage']
    expected_variables = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
    
    assert all(var in stats for var in expected_variables)
    assert all(metric in stats['TotalPremium'] for metric in expected_metrics)

def test_analyze_temporal_trends(temp_csv):
    """Test temporal trend analysis."""
    analyzer = InsuranceDataAnalyzer(temp_csv)
    trends = analyzer.analyze_temporal_trends()
    
    assert isinstance(trends, pd.DataFrame)
    assert 'TotalClaims_MoM_Change' in trends.columns
    assert 'TotalPremium_MoM_Change' in trends.columns

def test_analyze_vehicle_risk(temp_csv):
    """Test vehicle risk analysis."""
    analyzer = InsuranceDataAnalyzer(temp_csv)
    risk_analysis = analyzer.analyze_vehicle_risk()
    
    assert isinstance(risk_analysis, pd.DataFrame)
    assert ('TotalClaims', 'sum') in risk_analysis.columns
    assert ('LossRatio', 'mean') in risk_analysis.columns

def test_create_visualizations(temp_csv, tmp_path):
    """Test visualization creation."""
    analyzer = InsuranceDataAnalyzer(temp_csv)
    output_dir = tmp_path / 'figures'
    analyzer.create_visualizations(str(output_dir))
    
    expected_files = [
        'loss_ratio_by_province.png',
        'claims_vs_premium.png',
        'monthly_loss_ratio.png'
    ]
    
    for file in expected_files:
        assert (output_dir / file).exists() 