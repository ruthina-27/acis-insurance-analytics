# ACIS Insurance Analytics

This project contains data analytics solutions for AlphaCare Insurance Solutions (ACIS) to analyze risk patterns and optimize insurance strategies.

## Project Structure

```
acis-insurance-analytics/
├── data/                   # Data directory
├── notebooks/             # Jupyter notebooks for analysis
│   └── insurance_visualization.ipynb
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   └── visualization/    # Visualization modules
├── tests/                # Unit tests
├── reports/              # Analysis outputs
│   ├── figures/         # Generated visualizations
│   └── analysis/        # Analysis reports
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/acis-insurance-analytics.git
cd acis-insurance-analytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Analysis Components

1. **Exploratory Data Analysis (EDA)**
   - Loss Ratio Analysis by Province, VehicleType, and Gender
   - Distribution Analysis of Financial Variables
   - Temporal Trend Analysis
   - Vehicle Make/Model Risk Assessment

2. **Visualization**
   - Interactive Risk Score Heatmaps
   - Premium vs Claims Scatter Plots
   - Time Series Analysis with Trend Decomposition

## CI/CD Pipeline

Our GitHub Actions workflow includes:
- Code Quality (flake8, black)
- Testing (pytest)
- Coverage Reports

## Contributing

1. Create a new branch:
```bash
git checkout -b feature-name
```

2. Make changes and commit:
```bash
git add .
git commit -m "Descriptive commit message"
```

3. Create a pull request

## License

Proprietary - AlphaCare Insurance Solutions 