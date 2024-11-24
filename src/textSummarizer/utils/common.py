import os
import yaml
import requests
from box.exceptions import BoxValueError
from src.textSummarizer.logging import logger
from pydantic import validate_arguments
from box import ConfigBox
from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from datetime import datetime
import shutil
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from dask import delayed, compute
from dask.diagnostics import ProgressBar
# --- YAML and Directory Management Functions ---

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads yaml file and returns its contents.

    Args:
        path_to_yaml (Path): Path-like input to yaml file.

    Raises:
        ValueError: If yaml file is empty.
        Exception: For any other issues while reading the file.

    Returns:
        ConfigBox: ConfigBox object containing the yaml content.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file '{path_to_yaml}' loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty.")
    except Exception as e:
        raise e

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def create_directories(path_to_directories: List[Path], verbose: bool = True) -> None:
    """Creates a list of directories if they don't exist.

    Args:
        path_to_directories (list[Path]): List of directory paths.
        verbose (bool, optional): Flag to log directory creation. Defaults to True.
    """
    for path in path_to_directories:
        if not os.path.exists(path):  # Check if the directory already exists
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
        else:
            if verbose:
                logger.info(f"Directory already exists at: {path}")

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_size(path: Path) -> str:
    """Returns the size of a file in KB.

    Args:
        path (Path): Path to the file.

    Returns:
        str: Size of the file in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

# --- File Downloading Functions ---
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_pdf_metadata(filename: Path) -> Dict[str, Any]:
    """Extracts metadata from a PDF file.

    Args:
        filename (str): The path to the downloaded PDF file.

    Returns:
        Dict[str, Any]: Dictionary containing PDF metadata such as title, author, creation date, etc.
    """
    metadata = {}
    try:
        with open(filename, 'rb') as file:
            parser = PDFParser(file)
            document = PDFDocument(parser)
            metadata = document.info[0] if document.info else {}
            # Extract specific metadata fields
            title = metadata.get('Title', 'Unknown')
            author = metadata.get('Author', 'Unknown')
            creation_date = metadata.get('CreationDate', 'Unknown')
            modification_date = metadata.get('ModDate', 'Unknown')

            # Convert dates to readable format
            creation_date = creation_date[2:10] if isinstance(creation_date, str) else creation_date
            modification_date = modification_date[2:10] if isinstance(modification_date, str) else modification_date

            return {
                "title": title,
                "author": author,
                "creation_date": creation_date,
                "modification_date": modification_date,
            }
    except Exception as e:
        logger.error(f"Error reading metadata from PDF '{filename}': {e}")
        return {}

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def get_file_metadata(response: requests.Response, filename: Path) -> Dict[str, Any]:
    """Extracts metadata from the HTTP response and the local file system, including PDF metadata.

    Args:
        response (requests.Response): The response object after downloading the file.
        filename (str): The path to the downloaded file.

    Returns:
        Dict[str, Any]: Dictionary containing file metadata such as size, last modified date, and PDF-specific metadata.
    """
    file_size = os.path.getsize(filename)
    last_modified = response.headers.get('Last-Modified', None)

    if last_modified:
        last_modified = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S %Z')

    pdf_metadata = get_pdf_metadata(filename)

    metadata = {
        "file_name": filename.name,
        "file_size": f"{round(file_size / 1024, 2)} KB",
        "last_modified": last_modified,
        "download_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    metadata.update(pdf_metadata)
    return metadata

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def is_file_corrupted(filename: Path) -> bool:
    """Checks if the downloaded file is corrupted, with PDF-specific validation.

    Args:
        filename (Path): The path to the downloaded file.

    Returns:
        bool: True if file is corrupted, False otherwise.
    """
    try:
        if os.path.getsize(filename) == 0:
            raise Exception("File size is zero, likely corrupted.")

        with open(filename, 'rb') as file:
            parser = PDFParser(file)
            document = PDFDocument(parser)

            if not document.is_extractable:
                raise Exception("PDF text extraction is not allowed or the document is unreadable.")

        return False

    except Exception as e:
        logger.error(f"File '{filename}' is corrupted: {e}")
        return True

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def move_to_failed(filename: Path, failed_dir: Path) -> None:
    """Moves a corrupted file to a 'failed' directory for further inspection.

    Args:
        filename (Path): The path to the corrupted file.
        failed_dir (Path): The directory where failed files will be moved.
    """
    if not os.path.exists(failed_dir):
        os.makedirs(failed_dir)
    
    destination = failed_dir / filename.name
    shutil.move(str(filename), destination)
    logger.info(f"Moved corrupted file '{filename}' to '{failed_dir}'.")

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def download_file(url: Tuple[str, str], local_dir: Path, failed_dir: Path) -> Dict[str, Any]:
    """Downloads a single file from a given URL, handles corrupted files, and extracts its metadata.

    Args:
        url (Tuple[str, str]): Tuple with filename and URL.
        local_dir (Path): Path to store the downloaded files.
        failed_dir (Path): Directory for storing failed files.

    Returns:
        Dict[str, Any]: Dictionary containing metadata of the downloaded file, or error details if download fails.
    """
    os.environ['REQUESTS_CA_BUNDLE'] = 'C:\\Users\\Ikshvaku Rastogi\\Documents\\Documents\\Wasserstoff\\temp\\Lib\\site-packages\\certifi\\cacert.pem'
    
    try:
        response = requests.get(url[1], stream=True, timeout=15)  # Timeout set to 15 seconds
        response.raise_for_status()  # Raise exception for bad status codes

        # Define the file path
        filename = local_dir / (Path(url[0]).with_suffix(".pdf"))
        
        # Write the content to a file in chunks
        with open(filename, 'wb') as file:
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=str(filename), leave=False) as bar:
                for data in response.iter_content(chunk_size=4096):
                    file.write(data)
                    bar.update(len(data))

        # Check if the file is corrupted
        if is_file_corrupted(filename):
            move_to_failed(filename, failed_dir)
            logger.error(f"File '{filename}' is corrupted and moved to '{failed_dir}'.")
            return {"error": f"File '{filename}' is corrupted and moved to '{failed_dir}'."}

        # Get and return file metadata
        metadata = get_file_metadata(response, str(filename))
        logger.info(f"Successfully downloaded and saved file '{filename}' with metadata: {metadata}")
        return metadata

    except requests.exceptions.Timeout:
        error_message = f"Timeout error for URL: {url[1]}"
        logger.error(error_message)
        return {"error": error_message}

    except requests.exceptions.RequestException as e:
        error_message = f"Failed to download {url[1]}: {str(e)}"
        logger.error(error_message)
        return {"error": error_message}

    except Exception as e:
        error_message = f"Unexpected error downloading {url[0]}: {str(e)}"
        logger.error(error_message)
        return {"error": error_message}


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def download_files_in_parallel(urls: Dict[str, str], local_dir: Path, failed_dir: Path) -> List[Dict[str, Any]]:
    """Downloads multiple files in parallel using Dask and handles errors and metadata extraction.

    Args:
        urls (Dict[str, str]): Dictionary of filenames and URLs.
        local_dir (Path): Directory where the downloaded files should be saved.
        failed_dir (Path): Directory for storing failed files.

    Returns:
        List[Dict[str, Any]]: List of file metadata or error details.
    """
    
    downloaded_files = []

    # Create delayed tasks for each download
    tasks = [delayed(download_file)(url=(name, url), local_dir=local_dir, failed_dir=failed_dir) for name, url in urls.items()]

    # Progress bar using dask's diagnostics
    with ProgressBar():
        results = compute(*tasks, scheduler='threads')

    # Process results for logging and return
    for result in results:
        if "error" in result:
            logger.error(f"Error downloading file: {result['error']}")
        else:
            logger.info(f"File downloaded successfully. Metadata: {result}")
            downloaded_files.append(result)

    return downloaded_files
