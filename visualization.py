import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

df = pd.read_csv('lng_flare_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Time Series Overview
plt.figure(figsize=(15, 4))
plt.plot(df['timestamp'], df['total_flare_rate_m3_per_hour'], linewidth=0.15, alpha=0.7, color='#e74c3c')
plt.title('Flare Gas Rate - Historical Overview (2022 - 2025)', fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Flare Rate (m³/hr)')
plt.tight_layout()
plt.savefig('plot1_full_year_overview.png', dpi=150)
plt.close()

# 2. Daily Average
plt.figure(figsize=(15, 4))
daily_avg = df.groupby(df['timestamp'].dt.date)['total_flare_rate_m3_per_hour'].mean()
plt.plot(pd.to_datetime(daily_avg.index), daily_avg.values, linewidth=1.2, color='#3498db')
plt.title('Daily Average Flare Rate Trend', fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Avg Flare Rate (m³/hr)')
plt.tight_layout()
plt.savefig('plot2_daily_average.png', dpi=150)
plt.close()

# 3. Hourly Profile
plt.figure(figsize=(10, 4))
hourly_avg = df.groupby('hour')['total_flare_rate_m3_per_hour'].mean()
plt.bar(hourly_avg.index, hourly_avg.values, color='#2ecc71', alpha=0.7, edgecolor='black')
plt.title('Average Flare Rate by Hour of Day', fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Avg Rate (m³/hr)')
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig('plot3_hourly_pattern.png', dpi=150)
plt.close()

# 4. Monthly Volumetric Sums
plt.figure(figsize=(10, 4))
df['month_num'] = df['timestamp'].dt.month
monthly_total = df.groupby('month_num')['total_flare_rate_m3_per_hour'].sum() / 1000
plt.bar(range(1, 13), monthly_total.values, color=plt.cm.tab20(np.linspace(0, 1, 12)), edgecolor='black')
plt.title('Aggregated Monthly Total Flare Gas Volume', fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Total Gas Volume (1,000 m³)')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.savefig('plot4_monthly_totals.png', dpi=150)
plt.close()

# 5. Volumetric Share by Cause
plt.figure(figsize=(11, 5))
cause_cols = ['normal_operations_m3_per_hour', 'process_upset_m3_per_hour', 'equipment_maintenance_m3_per_hour',
              'startup_shutdown_m3_per_hour', 'emergency_relief_m3_per_hour', 'compressor_trip_m3_per_hour',
              'instrument_failure_m3_per_hour']
cause_totals = {c.replace('_m3_per_hour', '').replace('_', ' ').title(): df[c].sum() / 1000 for c in cause_cols}
cause_series = pd.Series(cause_totals).sort_values(ascending=True)
plt.barh(range(len(cause_series)), cause_series.values, color='#9b59b6', edgecolor='black')
plt.yticks(range(len(cause_series)), cause_series.index)
plt.title('Total Historical Flare Gas Volume by Cause', fontweight='bold')
plt.xlabel('Total Flare Gas (1,000 m³)')
plt.tight_layout()
plt.savefig('plot5_flare_by_cause.png', dpi=150)
plt.close()

# 6. Stacked Area Contribution Profile
plt.figure(figsize=(15, 5))
daily_stacked = df.set_index('timestamp')[cause_cols].resample('D').sum()
plt.stackplot(daily_stacked.index, [daily_stacked[c] for c in cause_cols], 
              labels=[c.replace('_m3_per_hour','').replace('_',' ').title() for c in cause_cols], alpha=0.85)
plt.title('Daily Flare Gas Contribution Profiles', fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Daily Total Flare Gas (m³)')
plt.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('plot6_stacked_causes.png', dpi=150)
plt.close()

# 7. Severity Distribution
plt.figure(figsize=(8, 4))
severity_counts = df['severity'].value_counts()
plt.bar(severity_counts.index, severity_counts.values, color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black', alpha=0.8)
plt.title('Distribution of Flare Events by Severity Level', fontweight='bold')
plt.xlabel('Severity')
plt.ylabel('Operating Hours')
plt.tight_layout()
plt.savefig('plot7_severity_distribution.png', dpi=150)
plt.close()

# 8. KDE Distribution Normal Operations
plt.figure(figsize=(10, 4))
normal_ops_data = df[df['normal_operations_m3_per_hour'] > 0]['normal_operations_m3_per_hour']
plt.hist(normal_ops_data, bins=50, color='#3498db', alpha=0.6, edgecolor='black', density=True)
normal_ops_data.plot(kind='kde', color='#e74c3c', linewidth=2, label='KDE')
plt.title('Distribution Density of Normal Operations Flare Rate', fontweight='bold')
plt.xlabel('Flare Rate (m³/hr)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('plot8_normal_ops_distribution.png', dpi=150)
plt.close()

# 9. Isolated Normal Operations Daily Tracking
plt.figure(figsize=(15, 4))
df.set_index('timestamp', inplace=True)
daily_normal_ops = df['normal_operations_m3_per_hour'].resample('D').mean()
df.reset_index(inplace=True)
plt.plot(daily_normal_ops.index, daily_normal_ops.values, color='#27ae60', linewidth=1, label='Isolated Baseline')
plt.title('Normal Operations Only: Daily Flaring Baseline (2022 - 2025)', fontweight='bold')
plt.xlabel('Timeline')
plt.ylabel('Gas Flare Volumetric Rate (m³/hr)')
plt.tight_layout()
plt.savefig('plot9_normal_ops_only.png', dpi=150)
plt.close()

# 10. Integrated Weekly Timeline
try:
    forecast_df = pd.read_csv('normal_ops_forecast_2026.csv')
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    
    df.set_index('timestamp', inplace=True)
    weekly_historical = df['normal_operations_m3_per_hour'].resample('W').mean().reset_index()
    df.reset_index(inplace=True)
    weekly_historical.columns = ['date', 'normal_ops_rate']
    recent_historical = weekly_historical[weekly_historical['date'] >= '2024-01-01']
    
    plt.figure(figsize=(16, 6.5))
    plt.plot(recent_historical['date'], recent_historical['normal_ops_rate'], 
             color='#2c3e50', linewidth=1.5, alpha=0.8, marker='o', markersize=3, label='Historical Weekly Baseline (2024-2025)')
    plt.plot(forecast_df['date'], forecast_df['forecasted_normal_ops_m3_per_hour'], 
             color='#e67e22', linewidth=2.5, marker='s', markersize=4, label='Weekly SARIMA Forecast (2026)')
    
    plt.axvspan(pd.Timestamp('2024-06-15'), pd.Timestamp('2024-07-31'), color='#e74c3c', alpha=0.2)
    plt.axvspan(pd.Timestamp('2025-06-15'), pd.Timestamp('2025-07-31'), color='#e74c3c', alpha=0.2)
    plt.axvspan(pd.Timestamp('2026-06-15'), pd.Timestamp('2026-07-31'), color='#e74c3c', alpha=0.2, label='Summer Thermal Peak Spans')
    
    plt.axvspan(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-02-15'), color='#3498db', alpha=0.2)
    plt.axvspan(pd.Timestamp('2025-01-01'), pd.Timestamp('2025-02-15'), color='#3498db', alpha=0.2)
    plt.axvspan(pd.Timestamp('2026-01-01'), pd.Timestamp('2026-02-15'), color='#3498db', alpha=0.2, label='Winter Operational Troughs')

    plt.title('Integrated Engineering Weekly Baseline Tracking vs Seasonal Forecast', fontsize=14, fontweight='bold')
    plt.xlabel('Production Timeline Horizon (Weekly Steps)')
    plt.ylabel('Normal Operations Volume Rate (m³/hr)')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    plt.savefig('plot10_integrated_sarima_forecast.png', dpi=300, bbox_inches='tight')
    print("✓ All 10 plots built and exported successfully.")
except FileNotFoundError:
    print("⚠️ Missing normal_ops_forecast_2026.csv. Run analysis.py first.")