import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

file_path = "aisdk-2024-03-19.csv"

## calculate the distance using the Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])  
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c ## return the calculated distance in kilometers

## function to detect location anomalies based on large jumps (distance above 100km threshold) in position. (100km was chosen because it's an really big jump compared to average vessel speed)
def detect_location_anomalies(chunk):
    ## store anomalies in a list
    anomalies = []
    ## group data by vessel identifier (MMSI) and process each seperately
    for _, vessel_data in tqdm(chunk.groupby('MMSI'), desc="Processing Location Anomalies"):
        vessel_data = vessel_data.sort_values(by='# Timestamp').reset_index(drop=True)
        ## create new collumns for past latitude and longitude
        vessel_data['prev_lat'] = vessel_data['Latitude'].shift(1)
        vessel_data['prev_lon'] = vessel_data['Longitude'].shift(1)
        ## distance between positions
        vessel_data['distance'] = calculate_distance(
            vessel_data['Latitude'], vessel_data['Longitude'],
            vessel_data['prev_lat'], vessel_data['prev_lon']
        )
        ## detect location anomalies based on large jumps (100km threshold) in position
        anomalies_df = vessel_data.loc[vessel_data['distance'] > 100, ['MMSI', '# Timestamp']]
        anomalies_df['Anomaly_Type'] = 'Large Location Jump'

        anomalies.append(anomalies_df.to_numpy())

    return np.vstack(anomalies) if anomalies else np.empty((0, 3))

## function to detect unrealistic speed anomalies (SOG above 50 knots threshold). (50 knots is almost impossible for modern day marine vessels)
def detect_speed_anomalies(chunk):
    ## I had problems with SOG containing NAs so I additionally used coerce with fillna(0)
    chunk['SOG'] = pd.to_numeric(chunk['SOG'], errors='coerce')
    ## store anomalies in a list
    anomalies = []
    ## group data by vessel identifier (MMSI) and process each seperately
    for _, vessel_data in tqdm(chunk.groupby('MMSI'), desc="Processing Speed Anomalies", total=len(chunk['MMSI'].unique())):
        ## detect unrealistic speed anomalies (SOG above 50 knots threshold)
        vessel_anomalies = vessel_data.loc[vessel_data['SOG'].fillna(0) > 50, ['MMSI', '# Timestamp']]

        vessel_anomalies['Anomaly_Type'] = 'Unrealistic Speed'
        
        if not vessel_anomalies.empty:
            anomalies.append(vessel_anomalies.to_numpy())
    
    return np.vstack(anomalies) if anomalies else np.empty((0, 3))

## funtion to process chunks in parallel with ProcessPoolExecutor
def process_chunk(chunk):
    with ProcessPoolExecutor() as executor:
        ## funtion to run location anomaly and speed anomaly detection in task parallel
        future_location = executor.submit(detect_location_anomalies, chunk)
        future_speed = executor.submit(detect_speed_anomalies, chunk)
        ## get results from both tasks
        location_anomalies = future_location.result()
        speed_anomalies = future_speed.result()
    ## combine results from both tasks
    return np.vstack((location_anomalies, speed_anomalies))

## funtion to load and process the AIS dataset in parallel using multiple workers (7 workers come from having an 8 core pc - 1 for the computer to function)
def load_and_process_data_parallel(file_path, chunk_size=100000, num_workers=7):
    ## chunk the data
    chunks = list(pd.read_csv(file_path, chunksize=chunk_size))
    ## process each chunk is parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_chunk, chunks), total=len(chunks), desc="Parallel Processing"))

    return np.vstack(results) if results else np.empty((0, 3))

## testing function to run spoofing detection using different numbers of workers
def run_spoofing_detection_varying_workers(file_path, worker_counts=[1, 2, 3, 4, 5, 6, 7]):
    results = []
    ## I used time library for execution time counts. Runs the funtion to load and process the AIS dataset in parallel with anomaly detection with different cpu counts (might've been a bit to much testing but I wanted to try it all out) 
    for num_workers in worker_counts:
        print(f"\nRunning with {num_workers} workers...")
        start_time = time.time()
        parallel_anomalies = load_and_process_data_parallel(file_path, num_workers=num_workers)
        parallel_time = time.time() - start_time
        print(f"Parallel Execution Time with {num_workers} workers: {parallel_time:.2f} seconds")

        results.append((num_workers, parallel_time))
    ## visualise results
    visualize_benchmarking(results)
    
    return results
## function to visualise the results in a bar chart
def visualize_benchmarking(results):
    ## extract worker counts and execution times
    worker_counts, times = zip(*results)
    ## bar chart specifics
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
    ## execute the code
    results = run_spoofing_detection_varying_workers(file_path)
