from src.textSummarizer.utils.common import (
    download_file, 
    download_files_in_parallel, 
    move_to_failed, 
    get_size
)
from src.textSummarizer.entity import DataIngestionConfig
from pathlib import Path
import os

class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_single_file(self, url: tuple) -> None:
        """
        Downloads a single file from the provided URL and saves it.
        Handles failed downloads by moving them to the failed directory.
        """
        print(f"Attempting to download file from URL: {url[1]}")
        
        # Download the file and retrieve its metadata or error details
        result = download_file(url=url, local_dir=self.config.local_data_file, failed_dir=self.config.failed_data_file)

        # Check if an error occurred during download
        if "error" in result:
            print(f"Error downloading file: {result['error']}")
        else:
            print(f"File downloaded successfully. Metadata: {result}")

    def download_multiple_files(self) -> None:
        """
        Downloads multiple files concurrently from a list of URLs.
        Utilizes the download_files_in_parallel function for parallel downloading.
        """
        print("Starting parallel download of files...")
        
        # Download multiple files concurrently
        results = download_files_in_parallel(urls=self.config.urls, local_dir=self.config.local_data_file, failed_dir=self.config.failed_data_file)
        import pandas as pd
        pd.DataFrame(results).to_csv('metadata.csv',index=False)


    def check_file_size(self, file_path: Path) -> str:
        """
        Checks the size of a file at the given path.
        """
        return get_size(file_path)

    def handle_failed_file(self, file_path: Path) -> None:
        """
        Moves a corrupted or failed file to the failed directory.
        """
        print(f"Moving failed file '{file_path}' to '{self.config.failed_data_file}'...")
        move_to_failed(filename=str(file_path), failed_dir=str(self.config.failed_data_file))

