import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def run_spoofing_detection_varying_workers(file_path, worker_counts=[1, 2, 3, 4, 5, 6, 7]):
    results = []

    for num_workers in worker_counts:
        print(f"\nRunning with {num_workers} workers...")
        start_time = time.time()
        parallel_anomalies = load_and_process_data_parallel(file_path, num_workers=num_workers)
        parallel_time = time.time() - start_time
        print(f"Parallel Execution Time with {num_workers} workers: {parallel_time:.2f} seconds")

        results.append((num_workers, parallel_time))

    visualize_benchmarking(results)
    
    return results

def visualize_benchmarking(results):
    worker_counts, times = zip(*results)

    plt.figure(figsize=(10, 6))
    plt.bar(worker_counts, times, color='teal')

    for i, time in enumerate(times):
        plt.text(worker_counts[i], time, f"{time:.2f}s", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('Execution Time vs Worker Count')
    plt.xlabel('Number of Workers')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(worker_counts)
    plt.show()

if __name__ == '__main__':

    results = run_spoofing_detection_varying_workers(file_path)