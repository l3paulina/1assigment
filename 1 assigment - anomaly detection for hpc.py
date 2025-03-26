import time
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
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
        ## run location anomaly and speed anomaly detection in task parallel
        future_location = executor.submit(detect_location_anomalies, chunk)
        future_speed = executor.submit(detect_speed_anomalies, chunk)
        ## get results from both tasks
        location_anomalies = future_location.result()
        speed_anomalies = future_speed.result()
    ## combine results from both tasks
    return np.vstack((location_anomalies, speed_anomalies))

## funtion to load and process the AIS dataset in parallel using multiple workers (for hpc - cpu count -1)
def load_and_process_data_parallel(file_path, chunk_size=100000, num_workers=mp.cpu_count()-1):
    ## chunk the data
    chunks = list(pd.read_csv(file_path, chunksize=chunk_size))
    ## process each chunk in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_chunk, chunks), total=len(chunks), desc="Parallel Processing"))

    return np.vstack(results) if results else np.empty((0, 3))

## ## funtion to process chunks in sequential
def load_and_process_data_sequential(file_path, chunk_size=100000):
    ## store anomalies in a list
    anomalies = []
    ## run the anomaly detection functions without parallization
    for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size), desc="Sequential Processing"):
        anomalies.append(process_chunk(chunk))

    return np.vstack(anomalies) if anomalies else np.empty((0, 3))

## function for spoofing detection and comparing execution times between parallel and sequential
def run_spoofing_detection(file_path):
    print("Running in parallel...")
    ## I used time library for execution time counts. Run the funtion to load and process the AIS dataset in parallel with anomaly detection
    start_time = time.time()
    parallel_anomalies = load_and_process_data_parallel(file_path)
    parallel_time = time.time() - start_time
    print(f"Parallel Execution Time: {parallel_time:.2f} seconds")

    print("\nRunning sequentially...")
    start_time = time.time()
    ## run the funtion to load and process the AIS dataset in sequential with anomaly detection
    sequential_anomalies = load_and_process_data_sequential(file_path)
    sequential_time = time.time() - start_time
    print(f"Sequential Execution Time: {sequential_time:.2f} seconds")

    ## calculate the speed up between sequantial and parallel
    speedup = sequential_time / parallel_time
    print(f"Speedup: {speedup:.2f}x")

    ## return anomaly counts and parallel time, sequential time and speed up
    return (
        parallel_anomalies[parallel_anomalies[:, 2] == 'Large Location Jump'],
        parallel_anomalies[parallel_anomalies[:, 2] == 'Unrealistic Speed'],
        parallel_time, sequential_time, speedup
    )

## function to identify the top 5 Ships with the most anomalies
def top_5_ships_with_most_outliers(location_anomalies, speed_anomalies):
    ## combine both location anomalies and speed anomalies into a single dataset
    location_df = pd.DataFrame(location_anomalies, columns=['MMSI', '# Timestamp', 'Anomaly_Type'])
    speed_df = pd.DataFrame(speed_anomalies, columns=['MMSI', '# Timestamp', 'Anomaly_Type'])
    all_anomalies = pd.concat([location_df, speed_df])
    ## count number of anomalies for each vessel (identified by MMSI)
    anomaly_counts = all_anomalies['MMSI'].value_counts()
    ## top 5 vessels with the highest anomaly counts
    top_5_mmsi = anomaly_counts.nlargest(5)
    ## print out results of the top 5
    print("Top 5 ships with the most anomalies:")
    for i, (mmsi, count) in enumerate(top_5_mmsi.items(), 1):
        print(f"{i}. MMSI {mmsi} with {count} anomalies")

    return top_5_mmsi

## function to visualize the anomalies
def visualize_anomalies(location_anomalies, speed_anomalies):
    ## convert the location anomalies and speed anomalies arrays into pandas dataframes (easier visualization)
    location_anomalies_df = pd.DataFrame(location_anomalies, columns=['MMSI', '# Timestamp', 'Anomaly_Type'])
    speed_anomalies_df = pd.DataFrame(speed_anomalies, columns=['MMSI', '# Timestamp', 'Anomaly_Type'])
    ## convert timestamps into datetime format for time-based analysis (I had problems with pd.to_datetime so coerce is there)
    location_anomalies_df['# Timestamp'] = pd.to_datetime(location_anomalies_df['# Timestamp'], errors='coerce')
    speed_anomalies_df['# Timestamp'] = pd.to_datetime(speed_anomalies_df['# Timestamp'], errors='coerce')

    anomaly_counts = {'Location Anomalies': len(location_anomalies), 'Speed Anomalies': len(speed_anomalies)}

    ## bar chart comparing the total number of location anomalies and speed anomalies
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(anomaly_counts.keys()), y=list(anomaly_counts.values()), palette=['blue', 'orange'])
    plt.title('Number of Anomalies Detected')
    plt.ylabel('Count')
    plt.xlabel('Anomaly Type')
    plt.show()

    ## histogram showing the distribution of location anomalies per vessel
    plt.figure(figsize=(10, 5))
    sns.histplot(location_anomalies_df['MMSI'].value_counts(), bins=30, kde=True, color='blue')
    plt.title('Distribution of Location Jump Anomalies per Vessel')
    plt.xlabel('Number of Location Jumps')
    plt.ylabel('Vessel Count')
    plt.show()

    ## time-series histogram visualizing when the anomalies occurred over time
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
    # execute code and print results
    location_anomalies, speed_anomalies, parallel_time, sequential_time, speedup = run_spoofing_detection(file_path)
    top_5_mmsi = top_5_ships_with_most_outliers(location_anomalies, speed_anomalies)
    visualize_anomalies(location_anomalies, speed_anomalies)
    print(f"\nSpeedup (Sequential vs Parallel): {speedup:.2f}x")
    print(f"Parallel Execution Time: {parallel_time:.2f} seconds")
    print(f"Sequential Execution Time: {sequential_time:.2f} seconds")