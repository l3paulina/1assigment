import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

file_path = "aisdk-2024-03-19.csv"

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])  
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def detect_location_anomalies(chunk):
    anomalies = []
    for _, vessel_data in tqdm(chunk.groupby('MMSI'), desc="Processing Location Anomalies"):
        vessel_data = vessel_data.sort_values(by='# Timestamp').reset_index(drop=True)

        vessel_data['prev_lat'] = vessel_data['Latitude'].shift(1)
        vessel_data['prev_lon'] = vessel_data['Longitude'].shift(1)

        vessel_data['distance'] = calculate_distance(
            vessel_data['Latitude'], vessel_data['Longitude'],
            vessel_data['prev_lat'], vessel_data['prev_lon']
        )
        anomalies_df = vessel_data.loc[vessel_data['distance'] > 100, ['MMSI', '# Timestamp']]
        anomalies_df['Anomaly_Type'] = 'Large Location Jump'

        anomalies.append(anomalies_df.to_numpy())

    return np.vstack(anomalies) if anomalies else np.empty((0, 3))

def detect_speed_anomalies(chunk):
    chunk['SOG'] = pd.to_numeric(chunk['SOG'], errors='coerce')
    anomalies = []
    for _, vessel_data in tqdm(chunk.groupby('MMSI'), desc="Processing Speed Anomalies", total=len(chunk['MMSI'].unique())):

        vessel_anomalies = vessel_data.loc[vessel_data['SOG'].fillna(0) > 50, ['MMSI', '# Timestamp']]

        vessel_anomalies['Anomaly_Type'] = 'Unrealistic Speed'
        
        if not vessel_anomalies.empty:
            anomalies.append(vessel_anomalies.to_numpy())
    
    return np.vstack(anomalies) if anomalies else np.empty((0, 3))

def process_chunk(chunk):
    with ProcessPoolExecutor() as executor:
        future_location = executor.submit(detect_location_anomalies, chunk)
        future_speed = executor.submit(detect_speed_anomalies, chunk)

        location_anomalies = future_location.result()
        speed_anomalies = future_speed.result()

    return np.vstack((location_anomalies, speed_anomalies))

def load_and_process_data_parallel(file_path, chunk_size=100000, num_workers=7):
    chunks = list(pd.read_csv(file_path, chunksize=chunk_size))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_chunk, chunks), total=len(chunks), desc="Parallel Processing"))

    return np.vstack(results) if results else np.empty((0, 3))

def load_and_process_data_sequential(file_path, chunk_size=100000):
    anomalies = []
    for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size), desc="Sequential Processing"):
        anomalies.append(process_chunk(chunk))

    return np.vstack(anomalies) if anomalies else np.empty((0, 3))

def run_spoofing_detection(file_path):
    print("Running in parallel...")
    start_time = time.time()
    parallel_anomalies = load_and_process_data_parallel(file_path)
    parallel_time = time.time() - start_time
    print(f"Parallel Execution Time: {parallel_time:.2f} seconds")

    print("\nRunning sequentially...")
    start_time = time.time()
    sequential_anomalies = load_and_process_data_sequential(file_path)
    sequential_time = time.time() - start_time
    print(f"Sequential Execution Time: {sequential_time:.2f} seconds")

    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")

    return (
        parallel_anomalies[parallel_anomalies[:, 2] == 'Large Location Jump'],
        parallel_anomalies[parallel_anomalies[:, 2] == 'Unrealistic Speed'],
        parallel_time, sequential_time, speedup
    )

def top_5_ships_with_most_outliers(location_anomalies, speed_anomalies):

    location_df = pd.DataFrame(location_anomalies, columns=['MMSI', '# Timestamp', 'Anomaly_Type'])
    speed_df = pd.DataFrame(speed_anomalies, columns=['MMSI', '# Timestamp', 'Anomaly_Type'])

    all_anomalies = pd.concat([location_df, speed_df])

    anomaly_counts = all_anomalies['MMSI'].value_counts()

    top_5_mmsi = anomaly_counts.nlargest(5)

    print("Top 5 ships with the most anomalies:")
    for i, (mmsi, count) in enumerate(top_5_mmsi.items(), 1):
        print(f"{i}. MMSI {mmsi} with {count} anomalies")

    return top_5_mmsi

def visualize_anomalies(location_anomalies, speed_anomalies):
    
    location_anomalies_df = pd.DataFrame(location_anomalies, columns=['MMSI', '# Timestamp', 'Anomaly_Type'])
    speed_anomalies_df = pd.DataFrame(speed_anomalies, columns=['MMSI', '# Timestamp', 'Anomaly_Type'])

    location_anomalies_df['# Timestamp'] = pd.to_datetime(location_anomalies_df['# Timestamp'], errors='coerce')
    speed_anomalies_df['# Timestamp'] = pd.to_datetime(speed_anomalies_df['# Timestamp'], errors='coerce')

    anomaly_counts = {'Location Anomalies': len(location_anomalies), 'Speed Anomalies': len(speed_anomalies)}
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(anomaly_counts.keys()), y=list(anomaly_counts.values()), palette=['blue', 'orange'])
    plt.title('Number of Anomalies Detected')
    plt.ylabel('Count')
    plt.xlabel('Anomaly Type')
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(location_anomalies_df['MMSI'].value_counts(), bins=30, kde=True, color='blue')
    plt.title('Distribution of Location Jump Anomalies per Vessel')
    plt.xlabel('Number of Location Jumps')
    plt.ylabel('Vessel Count')
    plt.show()

    plt.figure(figsize=(12, 5))
    sns.histplot(location_anomalies_df['# Timestamp'], bins=50, color='blue', label='Location Anomalies', alpha=0.6)
    sns.histplot(speed_anomalies_df['# Timestamp'], bins=50, color='orange', label='Speed Anomalies', alpha=0.6)
    plt.legend()
    plt.title('Anomaly Occurrences Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Anomalies')
    plt.xticks(rotation=45)
    plt.show()

if __name__ == '__main__':
    
    location_anomalies, speed_anomalies, parallel_time, sequential_time, speedup = run_spoofing_detection(file_path)
    top_5_mmsi = top_5_ships_with_most_outliers(location_anomalies, speed_anomalies)
    visualize_anomalies(location_anomalies, speed_anomalies)
    print(f"\nSpeedup (Sequential vs Parallel): {speedup:.2f}x")
    print(f"Parallel Execution Time: {parallel_time:.2f} seconds")
    print(f"Sequential Execution Time: {sequential_time:.2f} seconds")