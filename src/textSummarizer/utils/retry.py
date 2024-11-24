import time
import logging

def retry(max_retries=3, wait_time=2):
    """Retry decorator to handle retries for functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)  # Wait before retrying
                    else:
                        logging.error(f"Max retries exceeded for {func.__name__}")
                        return None  # Optionally return None if all retries fail
        return wrapper
    return decorator
