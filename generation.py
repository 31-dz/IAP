import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(31)
random.seed(31)

FLARE_CAUSES = {
    'normal_operations': {'baseline': 100, 'std': 8},
    'process_upset': {'avg_rate': 50, 'std': 15, 'probability': 0.15},
    'equipment_maintenance': {'avg_rate': 40, 'std': 12, 'probability': 0.10},
    'startup_shutdown': {'avg_rate': 200, 'std': 50, 'probability': 0.00},
    'emergency_relief': {'avg_rate': 80, 'std': 25, 'probability': 0.03},
    'compressor_trip': {'avg_rate': 120, 'std': 30, 'probability': 0.02},
    'instrument_failure': {'avg_rate': 60, 'std': 20, 'probability': 0.02}
}

def generate_flare_data(start_year=2022, end_year=2025):
    start_date = datetime(start_year, 1, 1, 0, 0, 0)
    end_date = datetime(end_year, 12, 31, 23, 0, 0)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    
    weekly_temp_modifiers = [
        0.85, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.95, 0.96,
        0.98, 1.00, 1.01, 1.03, 1.04, 1.06, 1.07, 1.09, 1.11, 1.13, 1.14, 1.16,
        1.17, 1.18, 1.18, 1.18, 1.17, 1.16, 1.14, 1.12, 1.10, 1.08, 1.06, 1.04,
        1.02, 1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85,
        0.85, 0.84, 0.84, 0.85
    ]
    
    shutdown_events = []
    for year in range(start_year, end_year + 1):
        shutdown_dates = [datetime(year, 2, 15), datetime(year, 4, 20), datetime(year, 7, 10), datetime(year, 10, 5)]
        for shutdown_date in shutdown_dates:
            duration_hours = random.randint(24, 72)
            shutdown_start = shutdown_date + timedelta(hours=random.randint(0, 23))
            for h in range(duration_hours):
                shutdown_events.append(shutdown_start + timedelta(hours=h))
                
    shutdown_events = set(shutdown_events)
    data = []
    baseline_rate = 100
    
    for dt in date_range:
        is_shutdown = dt in shutdown_events
        contributions = {cause: 0 for cause in FLARE_CAUSES.keys()}
        
        baseline_rate += np.random.normal(0, 0.5)
        baseline_rate = np.clip(baseline_rate, 80, 120)
        contributions['normal_operations'] = max(0, baseline_rate + np.random.normal(0, FLARE_CAUSES['normal_operations']['std']))
        
        hour = dt.hour
        if 0 <= hour < 6:
            contributions['normal_operations'] *= 0.95
        elif 8 <= hour < 16:
            contributions['normal_operations'] *= 1.02
        
        week_num = dt.isocalendar().week
        week_idx = min(week_num - 1, 51)
        contributions['normal_operations'] *= weekly_temp_modifiers[week_idx]
        
        if is_shutdown:
            shutdown_rate = np.random.normal(FLARE_CAUSES['startup_shutdown']['avg_rate'], FLARE_CAUSES['startup_shutdown']['std'])
            contributions['startup_shutdown'] = max(0, shutdown_rate)
        else:
            for cause in ['process_upset', 'equipment_maintenance', 'emergency_relief', 'compressor_trip', 'instrument_failure']:
                if random.random() < FLARE_CAUSES[cause]['probability']:
                    rate = np.random.normal(FLARE_CAUSES[cause]['avg_rate'], FLARE_CAUSES[cause]['std'])
                    contributions[cause] = max(0, rate)
        
        total_flare_rate = sum(contributions.values())
        severity = 'low' if total_flare_rate <= 200 else 'medium' if total_flare_rate <= 350 else 'high'
        dominant_cause = max(contributions.items(), key=lambda x: x[1])[0]
        
        data.append({
            'timestamp': dt,
            'total_flare_rate_m3_per_hour': round(total_flare_rate, 2),
            'normal_operations_m3_per_hour': round(contributions['normal_operations'], 2),
            'process_upset_m3_per_hour': round(contributions['process_upset'], 2),
            'equipment_maintenance_m3_per_hour': round(contributions['equipment_maintenance'], 2),
            'startup_shutdown_m3_per_hour': round(contributions['startup_shutdown'], 2),
            'emergency_relief_m3_per_hour': round(contributions['emergency_relief'], 2),
            'compressor_trip_m3_per_hour': round(contributions['compressor_trip'], 2),
            'instrument_failure_m3_per_hour': round(contributions['instrument_failure'], 2),
            'dominant_cause': dominant_cause,
            'severity': severity,
            'day_of_week': dt.strftime('%A'),
            'month': dt.strftime('%B'),
            'hour': dt.hour
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    flare_df = generate_flare_data(2022, 2025)
    
    file_name = input("Enter name for the output file (press Enter for default 'lng_flare_data.csv'): ").strip().strip("'\"")
    output_file = file_name if file_name else 'lng_flare_data.csv'
    
    flare_df.to_csv(output_file, index=False)
    print(f"✓ Data successfully saved to '{output_file}'")
