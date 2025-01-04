import os
import requests

# Repository details
repo_owner = "jaakkopasanen"
repo_name = "AutoEq"
branch = "master"  # Branch name
base_path = "measurements/oratory1990/data"
output_dir = "csv_files"  # Local directory to save .csv files

# Base URLs
api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"
raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/"

# Function to download files
def download_file(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {path}")
    else:
        print(f"Failed to download: {url}")

# Function to recursively fetch and download .csv files
def fetch_csv_files(api_url, folder_path=""):
    full_url = api_url + folder_path
    response = requests.get(full_url)
    if response.status_code == 200:
        items = response.json()
        for item in items:
            if item['type'] == 'dir':  # If it's a folder, recurse into it
                fetch_csv_files(api_url, item['path'])
            elif item['type'] == 'file' and item['name'].endswith('.csv'):
                # Construct the raw file URL
                file_url = raw_url + item['path']
                # Construct the local file path
                local_path = os.path.join(output_dir, item['path'])
                # Ensure the directory exists locally
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                # Download the file
                download_file(file_url, local_path)
    else:
        print(f"Failed to fetch directory: {folder_path} (Status: {response.status_code})")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Start fetching
fetch_csv_files(api_url, base_path)
