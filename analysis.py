import pandas as pd
import numpy as np
from scipy import stats
import warnings
from openpyxl.styles import Font, PatternFill, Alignment

warnings.filterwarnings('ignore')

def load_data(filename='lng_flare_data.csv'):
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def style_header(ws, row=1):
    for cell in ws[row]:
        if cell.value:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center")

def descriptive_statistics(df):
    total_rate = df['total_flare_rate_m3_per_hour']
    return pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
        'Value': [len(total_rate), total_rate.mean(), total_rate.median(), total_rate.std(),
                   total_rate.min(), total_rate.max(), stats.skew(total_rate), stats.kurtosis(total_rate)]
    })

def cause_specific_statistics(df):
    cause_cols = [
        'normal_operations_m3_per_hour', 'process_upset_m3_per_hour', 'equipment_maintenance_m3_per_hour',
        'startup_shutdown_m3_per_hour', 'emergency_relief_m3_per_hour', 'compressor_trip_m3_per_hour',
        'instrument_failure_m3_per_hour'
    ]
    results = []
    for col in cause_cols:
        data = df[col]
        active = data[data > 0]
        results.append({
            'Cause': col.replace('_m3_per_hour', '').replace('_', ' ').title(),
            'Total Volume (m³)': round(data.sum(), 2),
            'Active Hours': len(active),
            'Active %': f"{len(active)/len(df)*100:.2f}%",
            'Mean Active (m³/hr)': round(active.mean(), 2) if len(active) > 0 else 0,
            'Max Rate (m³/hr)': round(active.max(), 2) if len(active) > 0 else 0
        })
    return pd.DataFrame(results)

def run_weekly_year_span_sarima(df, output_csv='normal_ops_forecast_2026.csv'):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    df.set_index('timestamp', inplace=True)
    weekly_series = df['normal_operations_m3_per_hour'].resample('W').mean()
    df.reset_index(inplace=True)
    
    model = SARIMAX(weekly_series, order=(1, 1, 1), seasonal_order=(0, 1, 0, 52),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    
    forecast_steps = 52
    forecast_res = results.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start='2026-01-04', periods=forecast_steps, freq='W')
    raw_forecast = forecast_res.predicted_mean.values
    
    last_historical_val = weekly_series.values[-1]
    initial_gap = last_historical_val - raw_forecast[0]
    
    smoothed_forecast = np.copy(raw_forecast)
    for i in range(min(4, forecast_steps)):
        weight = 1.0 - ((i + 1) / 5)
        smoothed_forecast[i] += (initial_gap * weight)
        
    forecast_df = pd.DataFrame({
        'date': forecast_index,
        'forecasted_normal_ops_m3_per_hour': smoothed_forecast
    })
    
    forecast_df['forecasted_normal_ops_m3_per_hour'] = forecast_df['forecasted_normal_ops_m3_per_hour'].clip(lower=0)
    forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')
    forecast_df.to_csv(output_csv, index=False)
    return forecast_df

def main():
    df = load_data()
    forecast_df = run_weekly_year_span_sarima(df)
    output_file = 'lng_flare_statistical_analysis.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        pd.DataFrame({
            'Metric': ['Total Sample Records', 'Historical Date Range', 'Total Flaired Mass (m³)'],
            'Value': [len(df), f"{df['timestamp'].min()} to {df['timestamp'].max()}", df['total_flare_rate_m3_per_hour'].sum()]
        }).to_excel(writer, sheet_name='Summary', index=False)
        
        descriptive_statistics(df).to_excel(writer, sheet_name='Descriptive Stats', index=False)
        cause_specific_statistics(df).to_excel(writer, sheet_name='Cause Statistics', index=False)
        forecast_df.to_excel(writer, sheet_name='SARIMA 2026 Forecast', index=False)
        
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            style_header(ws)
            for column in ws.columns:
                max_len = max(len(str(cell.value or '')) for cell in column)
                ws.column_dimensions[column[0].column_letter].width = min(max_len + 3, 50)
                
    print("✓ Full report analysis compilation completed successfully.")

if __name__ == "__main__":
    main()