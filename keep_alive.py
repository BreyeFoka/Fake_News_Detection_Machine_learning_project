import requests
import logging
import os
import time
from datetime import datetime

# Setup logging to print to console (Render captures console output)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get the server URL from environment variable or use a default
SERVER_URL = os.getenv('RENDER_EXTERNAL_URL', 'http://localhost:5000')

def keep_alive():
    """Function to ping server and keep it awake"""
    while True:
        try:
            # Using the health check endpoint
            response = requests.get(f'{SERVER_URL}/')
            if response.status_code == 200:
                logging.info('Successfully pinged server - Server is awake')
                logging.info(f'Server response: {response.json()}')
            else:
                logging.error(f'Server returned status code: {response.status_code}')
        except requests.exceptions.RequestException as e:
            logging.error(f'Failed to connect to server: {str(e)}')
        
        # Wait for 10 minutes before next ping
        time.sleep(600)  # 600 seconds = 10 minutes

if __name__ == "__main__":
    logging.info('Starting keep-alive service...')
    keep_alive()
