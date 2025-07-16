import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pulp import *
from datetime import datetime, date
import io
import zipfile

# --- NEW: Import your ETL function ---
# Make sure ETL.py is in the same directory
from ETL import etl_long_to_wide

# Page configuration
st.set_page_config(
    page_title="Amos - ENERGYNEST",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Thermal Storage Optimization System - DA Market")
st.markdown("""
This application optimizes thermal storage operations to minimize energy costs by:
- Charging thermal storage during low electricity prices
- Using stored energy during high prices
- Considering grid charges and market restrictions
""")

# Add helpful guidance
st.info("👈 **Getting Started:** Use the sidebar to upload your data file and configure system parameters, then run the optimization below.")

# Sidebar for parameters
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload electricity price data (CSV)", type=['csv'])

# --- NEW: Add a checkbox to control the ETL process ---
transform_data = st.sidebar.checkbox(
    "Transform data (long to wide format)",
    value=True,
    help="Check this if your data has one row per timestamp (e.g., 'Date (CET)', 'Day Ahead Price'). Uncheck if your data is already wide (date, 00:00, 00:15...)."
)

st.sidebar.header("System Parameters")

# System parameters with defaults
Δt = st.sidebar.number_input("Time Interval (hours)", value=0.25, min_value=0.1, max_value=1.0, step=0.05)
Pmax_el = st.sidebar.number_input("Max Electrical Power (MW)", value=2.0, min_value=0.1, max_value=10.0, step=0.1)
Pmax_th = st.sidebar.number_input("Max Thermal Power (MW)", value=2.0, min_value=0.1, max_value=10.0, step=0.1)
Smax = st.sidebar.number_input("Max Storage Capacity (MWh)", value=8.0, min_value=1.0, max_value=50.0, step=0.5)
SOC_min = st.sidebar.number_input("Min Storage Level (MWh)", value=0.0, min_value=0.0, max_value=5.0, step=0.5)
η = st.sidebar.number_input("Charging Efficiency", value=0.95, min_value=0.7, max_value=1.0, step=0.05)
self_discharge_daily = st.sidebar.number_input("Self-Discharge Rate (% per day)", value=3.0, min_value=0.0 ,max_value=20.0, step=0.1, help="Daily percentage of stored energy lost due to standing thermal losses.")
D_th = st.sidebar.number_input("Thermal Demand (MW)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)
C_grid = st.sidebar.number_input("Grid Charges (€/MWh)", value=30.0, min_value=0.0, max_value=100.0, step=1.0)
C_gas = st.sidebar.number_input("Gas Price (€/MWh)", value=65.0, min_value=10.0, max_value=200.0, step=1.0)
boiler_efficiency_pct = st.sidebar.number_input( "Gas Boiler Efficiency (%)", value=90.0, min_value=50.0, max_value=100.0, step=1.0, help="Efficiency of the gas boiler in converting gas fuel to thermal energy.")
boiler_efficiency = boiler_efficiency_pct / 100.0
terminal_value = st.sidebar.number_input("Terminal Value (€/MWh)", value=65.0, min_value=10.0, max_value=200.0, step=1.0)

# Holiday dates
st.sidebar.header("Holiday Configuration")
default_holidays = [
    '2024-01-01', '2024-03-29', '2024-04-01', '2024-05-01', '2024-05-09',
    '2024-05-10', '2024-05-20', '2024-05-30', '2024-05-31', '2024-10-01',
    '2024-10-04', '2024-11-01', '2024-12-23', '2024-12-24', '2024-12-25',
    '2024-12-26', '2024-12-27', '2024-12-28', '2024-12-29', '2024-12-30',
    '2024-12-31'
]
holiday_input = st.sidebar.text_area("Holiday Dates (one per line, YYYY-MM-DD)",
                                     value='\n'.join(default_holidays),
                                     height=150)
holiday_dates = [date.strip() for date in holiday_input.split('\n') if date.strip()]
holiday_set = set(holiday_dates)

# Hochlast periods
st.sidebar.header("Peak Period Restrictions")
hochlast_morning = st.sidebar.checkbox("Morning Peak (8-10 AM)", value=True)
hochlast_evening = st.sidebar.checkbox("Evening Peak (6-8 PM)", value=True)

# Build hochlast intervals
hochlast_intervals = set()
if hochlast_morning:
    hochlast_intervals.update(range(32, 40))  # 8-10 AM
if hochlast_evening:
    hochlast_intervals.update(range(72, 80))  # 6-8 PM

# --- MAIN LOGIC CHANGE ---
if uploaded_file is not None:
    df = None

    if transform_data:
        st.info("Uploaded file detected. Running ETL transformation...")
        with st.spinner("Transforming data from long to wide format..."):
            try:
                # Call the ETL function with the uploaded file object
                df_wide = etl_long_to_wide(
                    input_source=uploaded_file,
                    datetime_column_name='Date (CET)',
                    value_column_name='Day Ahead Price'
                )
                st.success("✅ ETL transformation successful!")
                df = df_wide # Use the transformed dataframe

            except Exception as e:
                # Catch the error raised from the ETL function and display it
                st.error(f"❌ ETL process failed: {e}")
                st.info("Please check your file format. Ensure the header row is correct and column names match the expected input.")
                st.stop() # Stop the app from running further

    else:
        # If the box is unchecked, load the data directly
        try:
            st.info("Loading data directly in wide format.")
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Failed to load the CSV file: {e}")
            st.stop()

    # From here, the app continues only if 'df' is a valid DataFrame
    if df is not None:
        try:
            # --- NEW: DATE RANGE SELECTION WIDGETS ---
            st.sidebar.header("🗓️ Date Range Filter")

            # Ensure the 'date' column is in datetime format for min/max operations
            df['date_obj'] = pd.to_datetime(df['date'])

            min_date = df['date_obj'].min().date()
            max_date = df['date_obj'].max().date()

            # Create two columns for side-by-side date pickers
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

            # Validate the date range
            if start_date > end_date:
                st.sidebar.error("Error: Start date cannot be after end date.")
                st.stop()

            # Filter the dataframe based on the selected date range
            mask = (df['date_obj'] >= pd.to_datetime(start_date)) & (df['date_obj'] <= pd.to_datetime(end_date))
            df_filtered = df.loc[mask].drop(columns=['date_obj'])  # Drop the temporary datetime column

            st.success(f"✅ Loaded {len(df_filtered)} days of data within the selected range ({start_date} to {end_date})")

            # Display data preview
            with st.expander("📊 Data Preview (filtered)"):
                st.dataframe(df_filtered.head(9))

                # Show data statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Days", len(df_filtered))
                with col2:
                    time_cols = [col for col in df_filtered.columns if col != 'date']
                    st.metric("Time Intervals/Day", len(time_cols))
                with col3:
                    if len(time_cols) > 0:
                        avg_price = df_filtered[time_cols].mean().mean()
                        st.metric("Avg Price (€/MWh)", f"{avg_price:.2f}")

            # Data cleaning
            st.header("Data Cleaning")
            with st.spinner("Cleaning data..."):
                for col in df_filtered.columns:
                    if col != 'date':
                        df_filtered[col] = df_filtered[col].replace([np.inf, -np.inf], np.nan)
                        df_filtered[col] = df_filtered[col].interpolate(method='linear', limit_direction='both')
                        if df_filtered[col].isna().any():
                            col_median = df_filtered[col].median()
                            df_filtered[col] = df_filtered[col].fillna(col_median)

            st.success("✅ Data cleaning completed")

            # Get time columns
            time_cols = [col for col in df_filtered.columns if col != 'date']

            # Functions for optimization
            def build_thermal_model(prices, soc0, η_self, boiler_eff, is_holiday=False):
                """Build optimization model for thermal storage system"""
                T = len(prices)
                model = LpProblem("Thermal_Storage", LpMinimize)

                # Decision variables
                p_el = LpVariable.dicts("p_el", range(T), lowBound=0, upBound=Pmax_el)
                p_th = LpVariable.dicts("p_th", range(T), lowBound=0, upBound=Pmax_th)
                p_gas = LpVariable.dicts("p_gas", range(T), lowBound=0) # Represents THERMAL output from boiler
                soc = LpVariable.dicts("soc", range(T), lowBound=SOC_min, upBound=Smax)

                # Objective: minimize total cost
                # --- FIX: Rearrange the expression to avoid dividing an LpVariable ---
                # Instead of `C_gas * (p_gas[t] / boiler_eff)`, we use `(C_gas / boiler_eff) * p_gas[t]`
                # This is mathematically identical but compatible with PuLP.
                model += lpSum([
                    (prices[t] + C_grid) * p_el[t] * Δt +
                    (C_gas / boiler_eff) * p_gas[t] * Δt
                    for t in range(T)
                ]) - terminal_value * soc[T-1]

                for t in range(T):
                    # Thermal balance
                    model += p_th[t] + p_gas[t] == D_th

                    # Hochlast constraint
                    if t in hochlast_intervals and not is_holiday:
                        model += p_el[t] == 0

                    # Storage dynamics
                    if t == 0:
                        model += soc[t] == soc0 * η_self + η * p_el[t] * Δt - p_th[t] * Δt
                    else:
                        model += soc[t] == soc[t-1] * η_self + η * p_el[t] * Δt - p_th[t] * Δt

                return model, p_el, p_th, p_gas, soc

            # Run optimization
            if st.button("🚀 Run Optimization", type="primary"):
                # Clear previous results
                if 'results' in st.session_state:
                    del st.session_state['results']
                if 'all_trades' in st.session_state:
                    del st.session_state['all_trades']
                if 'gas_baseline' in st.session_state:
                    del st.session_state['gas_baseline']

                progress_bar = st.progress(0)
                status_text = st.empty()

                soc0 = SOC_min
                results = []
                all_trades = []
                # Gas baseline correctly accounts for boiler efficiency
                gas_baseline = (D_th * 24 * C_gas) / boiler_efficiency

                η_self = (1 - self_discharge_daily / 100) ** (Δt / 24)

                # Process each day in the filtered dataframe
                for idx, (_, row) in enumerate(df_filtered.iterrows()):
                    progress_bar.progress((idx + 1) / len(df_filtered))
                    status_text.text(f"Processing day {idx + 1}/{len(df_filtered)}: {row['date']}")

                    day = row['date']
                    prices = row[time_cols].values

                    if len(prices) != 96:
                        continue

                    is_holiday = day in holiday_set

                    # Build and solve model
                    model, p_el, p_th, p_gas, soc = build_thermal_model(prices, soc0, η_self, boiler_efficiency, is_holiday)
                    status = model.solve(PULP_CBC_CMD(msg=False))

                    if status == 1:  # successful optimization
                        actual_elec_cost = sum((prices[t] + C_grid) * p_el[t].value() * Δt for t in range(96))
                        # Gas cost calculation (using float values) is correct
                        actual_gas_cost = sum(C_gas * (p_gas[t].value() / boiler_efficiency) * Δt for t in range(96))
                        soc_end = soc[95].value()

                        actual_total_cost = actual_elec_cost + actual_gas_cost - terminal_value * soc_end
                        savings = gas_baseline - actual_total_cost

                        elec_energy = sum([p_el[t].value() * Δt for t in range(96)])
                        # Gas fuel energy calculation (using float values) is correct
                        gas_fuel_energy = sum([(p_gas[t].value() / boiler_efficiency) * Δt for t in range(96)])

                        # Store detailed trading data
                        for t in range(96):
                            interval_hour = t // 4
                            interval_min = (t % 4) * 15
                            time_str = f"{interval_hour:02d}:{interval_min:02d}:00"

                            gas_cost_interval_val = C_gas * (p_gas[t].value() / boiler_efficiency) * Δt
                            elec_cost_interval_val = (prices[t] + C_grid) * p_el[t].value() * Δt

                            trade_record = {
                                'date': day,
                                'time': time_str,
                                'interval': t,
                                'da_price': prices[t],
                                'total_elec_cost': prices[t] + C_grid,
                                'p_el_heater': p_el[t].value(),
                                'p_th_discharge': p_th[t].value(),
                                'p_gas_backup': p_gas[t].value(), # Thermal output
                                'soc': soc[t].value(),
                                'elec_cost_interval': elec_cost_interval_val,
                                'gas_cost_interval': gas_cost_interval_val,
                                'total_cost_interval': elec_cost_interval_val + gas_cost_interval_val,
                                'is_hochlast': t in hochlast_intervals and not is_holiday,
                                'is_holiday': is_holiday,
                                'is_charging': p_el[t].value() > 0.01,
                                'is_discharging': p_th[t].value() > 0.01,
                                'using_gas': p_gas[t].value() > 0.01
                            }
                            all_trades.append(trade_record)

                        # Update SOC for next day
                        soc0 = soc_end

                        # Store daily results
                        results.append({
                            "day": day,
                            "cost": actual_total_cost,
                            "savings": savings,
                            "soc_end": soc_end,
                            "elec_energy": elec_energy,
                            "gas_energy": gas_fuel_energy, # Store gas FUEL energy
                            "is_holiday": is_holiday
                        })

                progress_bar.progress(1.0)
                status_text.text("✅ Optimization completed!")

                # Store results in session state
                st.session_state['results'] = results
                st.session_state['all_trades'] = all_trades
                st.session_state['gas_baseline'] = gas_baseline

        except Exception as e:
            st.error(f"❌ An error occurred after loading the data: {str(e)}")
            st.info("Please ensure your CSV file is in the expected format.")
            st.stop()

        # Display results (check session state first)
        if 'results' in st.session_state and st.session_state['results']:
            results = st.session_state['results']
            all_trades = st.session_state['all_trades']
            gas_baseline = st.session_state['gas_baseline']

            # Add clear results button
            col1, col2 = st.columns([3, 1])
            with col1:
                st.header("📊 Results Summary")
            with col2:
                if st.button("🗑️ Clear Results"):
                    del st.session_state['results']
                    del st.session_state['all_trades']
                    del st.session_state['gas_baseline']
                    st.rerun()

            # Calculate summary statistics
            avg_cost = np.mean([r['cost'] for r in results])
            avg_savings = np.mean([r['savings'] for r in results])
            total_savings = sum([r['savings'] for r in results])
            avg_elec = np.mean([r['elec_energy'] for r in results])
            avg_gas = np.mean([r['gas_energy'] for r in results]) # This is now avg gas FUEL
            savings_pct = (avg_savings / gas_baseline) * 100

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Days Analyzed", len(results))
            with col2:
                st.metric("Average Daily Savings", f"€{avg_savings:.0f}", f"{savings_pct:.1f}%")
            with col3:
                st.metric("Total Savings", f"€{total_savings:.0f}")
            with col4:
                st.metric("Gas Baseline", f"€{gas_baseline:.0f}/day")

            # Thermal contribution mix calculation
            thermal_from_elec = avg_elec * η
            thermal_from_gas = avg_gas * boiler_efficiency
            total_thermal_delivered = thermal_from_elec + thermal_from_gas
            
            if total_thermal_delivered > 0:
                elec_percentage = (thermal_from_elec / total_thermal_delivered) * 100
                gas_percentage = (thermal_from_gas / total_thermal_delivered) * 100
            else:
                elec_percentage, gas_percentage = 0, 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Thermal from Electricity", f"{elec_percentage:.1f}%")
            with col2:
                st.metric("Thermal from Gas", f"{gas_percentage:.1f}%")

            # Break-even analysis with boiler efficiency
            cost_gas_per_mwh_th = C_gas / boiler_efficiency
            break_even_price = (cost_gas_per_mwh_th * η) - C_grid
            st.info(f"**Break-even electricity price:** {break_even_price:.1f} €/MWh")

            # Best and worst days
            best_day = max(results, key=lambda x: x['savings'])
            worst_day = min(results, key=lambda x: x['savings'])

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Best day:** {best_day['day']} (€{best_day['savings']:.2f} saved)")
            with col2:
                st.warning(f"**Worst day:** {worst_day['day']} (€{worst_day['savings']:.2f} saved)")

            # Visualizations
            st.header("📈 Visualizations")

            # Daily savings chart
            results_df = pd.DataFrame(results)
            results_df['date'] = pd.to_datetime(results_df['day'])

            fig1 = px.line(results_df, x='date', y='savings',
                         title='Daily Savings Over Time',
                         labels={'savings': 'Savings (€)', 'date': 'Date'})
            fig1.add_hline(y=avg_savings, line_dash="dash",
                          annotation_text=f"Average: €{avg_savings:.2f}")
            st.plotly_chart(fig1, use_container_width=True)

            # Cumulative savings chart
            results_df['cumulative_savings'] = results_df['savings'].cumsum()
            fig2 = px.area(results_df, x='date', y='cumulative_savings',
                           title='Cumulative Savings Over Time',
                           labels={'cumulative_savings': 'Total Savings (€)', 'date': 'Date'})
            st.plotly_chart(fig2, use_container_width=True)

            # Energy mix chart (showing INPUT energies)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=results_df['date'], y=results_df['elec_energy'],
                                    mode='lines', name='Electricity Input', fill='tonexty'))
            fig3.add_trace(go.Scatter(x=results_df['date'], y=results_df['gas_energy'],
                                    mode='lines', name='Gas Fuel Input', fill='tozeroy'))
            fig3.update_layout(title='Daily Energy Input Mix',
                              xaxis_title='Date', yaxis_title='Energy (MWh)')
            st.plotly_chart(fig3, use_container_width=True)

            # Sample day analysis
            st.header("Sample Day Analysis")

            with st.expander("🔍 Detailed Day Analysis"):
                sample_day = st.selectbox("Select a day to analyze:",
                                        options=results_df['day'].tolist())

                if sample_day:
                    day_trades = pd.DataFrame([t for t in all_trades if t['date'] == sample_day])

                    if not day_trades.empty:
                        # Create subplot with multiple y-axes
                        fig4 = make_subplots(
                            rows=3, cols=1,
                            subplot_titles=('Electricity Price & Storage Operations',
                                          'State of Charge', 'Cost Breakdown'),
                            vertical_spacing=0.1,
                            specs=[[{"secondary_y": True}],
                                    [{"secondary_y": False}],
                                    [{"secondary_y": False}]]
                        )

                        # Price and operations
                        fig4.add_trace(go.Scatter(x=day_trades['time'], y=day_trades['da_price'],
                                                name='DA Price', line=dict(color='blue')), row=1, col=1, secondary_y=False)
                        fig4.add_trace(go.Scatter(x=day_trades['time'], y=day_trades['p_el_heater'],
                                                name='Charging', line=dict(color='green', dash='dot')), row=1, col=1, secondary_y=True)
                        fig4.add_trace(go.Scatter(x=day_trades['time'], y=day_trades['p_th_discharge'],
                                                name='Discharging', line=dict(color='red', dash='dot')), row=1, col=1, secondary_y=True)
                        # SOC
                        fig4.add_trace(go.Scatter(x=day_trades['time'], y=day_trades['soc'],
                                                name='SOC', line=dict(color='orange')), row=2, col=1)

                        # Costs
                        fig4.add_trace(go.Scatter(x=day_trades['time'], y=day_trades['elec_cost_interval'],
                                                name='Elec Cost', line=dict(color='blue')), row=3, col=1)
                        fig4.add_trace(go.Scatter(x=day_trades['time'], y=day_trades['gas_cost_interval'],
                                                name='Gas Cost', line=dict(color='red')), row=3, col=1)

                        # Update layout with better x-axis formatting
                        fig4.update_layout(
                            height=800,
                            title_text=f"Detailed Analysis for {sample_day}"
                        )
                        fig4.update_yaxes(title_text="Price (€/MWh)", row=1, col=1, secondary_y=False)
                        fig4.update_yaxes(title_text="Power (MW)", row=1, col=1, secondary_y=True, showgrid=False)
                        fig4.update_yaxes(title_text="Storage (MWh)", row=2, col=1)
                        fig4.update_yaxes(title_text="Cost (€)", row=3, col=1)
                        # Format x-axis to show time labels nicely - show only every 4th interval (hourly)
                        hourly_times = day_trades[day_trades['interval'] % 4 == 0]['time'].tolist()
                        fig4.update_xaxes(
                            tickmode='array',
                            tickvals=hourly_times,
                            ticktext=[t.split(':')[0] + ':00' for t in hourly_times],
                            tickangle=0
                        )

                        st.plotly_chart(fig4, use_container_width=True)

            # Download results
            st.header("💾 Download Results")

            # Create download files
            if all_trades:
                trades_df = pd.DataFrame(all_trades)

                # Create a ZIP file with all results
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add trades CSV
                    trades_csv = trades_df.to_csv(index=False)
                    zip_file.writestr('thermal_storage_trades.csv', trades_csv)

                    # Add daily summary CSV
                    daily_csv = results_df.to_csv(index=False)
                    zip_file.writestr('thermal_storage_daily.csv', daily_csv)

                    # Add boiler efficiency to parameters summary
                    params_text = f"""Thermal Storage Optimization Parameters
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

System Parameters:
- Time Interval: {Δt} hours
- Max Electrical Power: {Pmax_el} MW
- Max Thermal Power: {Pmax_th} MW
- Max Storage Capacity: {Smax} MWh
- Min Storage Level: {SOC_min} MWh
- Charging Efficiency: {η}
- Self-Discharge Rate: {self_discharge_daily} % per day
- Thermal Demand: {D_th} MW
- Grid Charges: {C_grid} €/MWh
- Gas Price: {C_gas} €/MWh
- Gas Boiler Efficiency: {boiler_efficiency_pct} %
- Terminal Value: {terminal_value} €/MWh

Results Summary:
- Days Analyzed: {len(results)}
- Average Daily Savings: €{avg_savings:.2f} ({savings_pct:.1f}%)
- Total Savings: €{total_savings:.2f}
- Thermal Contribution from Electricity: {elec_percentage:.1f}%
- Break-even Price: {break_even_price:.1f} €/MWh
"""
                    zip_file.writestr('parameters_and_summary.txt', params_text)

                zip_buffer.seek(0)

                st.download_button(
                    label="📥 Download All Results (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"thermal_storage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

                # Individual file downloads
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📊 Download Detailed Trades (CSV)",
                        data=trades_df.to_csv(index=False),
                        file_name='thermal_storage_trades.csv',
                        mime='text/csv'
                    )
                with col2:
                    st.download_button(
                        label="📅 Download Daily Summary (CSV)",
                        data=results_df.to_csv(index=False),
                        file_name='thermal_storage_daily.csv',
                        mime='text/csv'
                    )
        else:
            # Show message when no results are available
            st.info("🔍 Run optimization to see results and download options.")

else:
    st.info("👈 Please upload a CSV file using the sidebar to begin.")

    with st.expander("📋 Expected Data Format Guide"):
        st.markdown("""
        This app can handle two data formats. Select the correct option in the sidebar.

        ---

        #### 1. Long Format (when "Transform data" is checked)
        This is the preferred format for upload. The ETL process will convert it automatically.
        - A column with datetime information (e.g., `Date (CET)`).
        - A column with the price/value you want to use (e.g., `Day Ahead Price`).
        - Each row represents a single time step.

        **Example (`idprices-epexshort.csv`):**
        ```
        Date (CET),Day Ahead Price,...
        [01/01/2024 00:00],0.10,...
        [01/01/2024 00:15],0.10,...
        ```

        ---

        #### 2. Wide Format (when "Transform data" is unchecked)
        If your data is already processed into this format.
        - A 'date' column with dates in YYYY-MM-DD format.
        - 96 columns with electricity prices for each 15-minute interval (e.g., `00:00:00`, `00:15:00`, etc.).

        **Example:**
        ```
        date,00:00:00,00:15:00,...,23:45:00
        2024-01-01,45.2,43.1,...,52.3
        2024-01-02,38.7,36.2,...,48.9
        ```
        """)

# Footer
st.markdown("---")