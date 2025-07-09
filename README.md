# 🔥 Thermal Storage Optimization - Streamlit App

This is an interactive web application that optimizes thermal storage operations to minimize energy costs by intelligently managing the charging and discharging of thermal storage systems.

## Features

- **Interactive Parameter Configuration**: Adjust system parameters like storage capacity, power limits, efficiency, and pricing via sidebar controls
- **File Upload**: Upload your electricity price data in CSV format
- **Real-time Optimization**: Run optimization algorithms with live progress tracking
- **Comprehensive Visualizations**: View daily savings, storage state of charge, energy mix, and detailed day-by-day analysis
- **Results Export**: Download detailed results as CSV files or complete ZIP packages
- **Holiday Management**: Configure holiday dates that bypass peak period restrictions
- **Peak Period Restrictions**: Set up Hochlast periods where charging is restricted

## Installation

1. **Clone or download the files** to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run streamlit_thermal_storage.py
   ```

4. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## Data Format

Your CSV file should contain:
- A `date` column with dates in YYYY-MM-DD format
- 96 columns with electricity prices (one for each 15-minute interval)
- Column names can be anything except 'date'

Example structure:
```
date,00:00,00:15,00:30,...,23:45
2024-01-01,45.2,43.1,41.5,...,52.3
2024-01-02,38.7,36.2,34.8,...,48.9
```

## How to Use

1. **Configure Parameters**: Use the sidebar to adjust system parameters like:
   - Storage capacity and power limits
   - Efficiency and demand settings
   - Pricing parameters (grid charges, gas price)
   - Holiday dates and peak period restrictions

2. **Upload Data**: Upload your electricity price CSV file using the file uploader

3. **Run Optimization**: Click the "🚀 Run Optimization" button to start the analysis

4. **View Results**: Explore the comprehensive results including:
   - Summary statistics and key metrics
   - Interactive visualizations
   - Detailed day-by-day analysis
   - Best and worst performing days

5. **Download Results**: Export your results as CSV files or complete ZIP packages

## System Parameters

- **Time Interval**: Duration of each time step (default: 0.25 hours = 15 minutes)
- **Max Electrical/Thermal Power**: Maximum power capacity for charging and discharging
- **Storage Capacity**: Maximum energy storage capacity in MWh
- **Efficiency**: Charging efficiency of the thermal storage system
- **Thermal Demand**: Constant thermal demand that must be met
- **Grid Charges**: Additional charges for electricity from the grid
- **Gas Price**: Cost of gas backup energy
- **Terminal Value**: Value assigned to stored energy at the end of analysis

## Peak Period Restrictions (Hochlast)

The system supports peak period restrictions where charging is not allowed:
- **Morning Peak**: 8:00 AM - 10:00 AM (intervals 32-39)
- **Evening Peak**: 6:00 PM - 8:00 PM (intervals 72-79)

These restrictions are automatically bypassed on configured holiday dates.

## Optimization Algorithm

The application uses linear programming (PuLP) to solve the thermal storage optimization problem by:
1. Minimizing total energy costs (electricity + gas - terminal value)
2. Ensuring thermal demand is always met (storage discharge + gas backup)
3. Respecting storage capacity and power limits
4. Applying peak period restrictions when applicable
5. Maintaining storage energy balance across time periods

## Output Files

The application generates several output files:
- **thermal_storage_trades.csv**: Detailed 15-minute interval data
- **thermal_storage_daily.csv**: Daily summary statistics
- **parameters_and_summary.txt**: Configuration parameters and results summary

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- PuLP

## Support

For issues or questions, please check that:
1. Your CSV file follows the expected format
2. All required dependencies are installed
3. The PuLP solver is working correctly

---

Built with ❤️ using Streamlit | Thermal Storage Optimization System 