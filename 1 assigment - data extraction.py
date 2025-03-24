import os
import requests
from zipfile import ZipFile
import pandas as pd

## download the AIS data and extract it
def download_and_extract_zip(url, extract_to="ais_data"):
    zip_path = "ais_data.zip"
    
    print(f"Downloading data from {url}...")
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    
    print("Extracting data...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(zip_path)
    return extract_to

# format the data into a csv
def load_and_save_data_to_csv(folder, output_file="ais_data.csv"):

    file_path = os.path.join(folder, "aisdk-2024-03-19.csv") 
    print(f"Loading data from {file_path}...")

    data = pd.read_csv(file_path)
    print(f"Saving data to {output_file}...")

    data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

file_url = 'https://web.ais.dk/aisdata/aisdk-2024-03-19.zip'
extracted_folder = download_and_extract_zip(file_url)
csv_file = load_and_save_data_to_csv("ais_data", output_file="ais_data.csv")
