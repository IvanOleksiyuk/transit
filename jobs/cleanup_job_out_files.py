import os
import time
from datetime import datetime, timedelta

def delete_old_files(directory):
    # Get the current time
    now = time.time()
    # Calculate the time 24 hours ago
    cutoff = now - (24 * 60 * 60)
    
    # Walk through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # Get the last modification time of the file
            file_mtime = os.path.getmtime(file_path)
            # If the file is older than 24 hours, delete it
            if file_mtime < cutoff:
                print(f"Deleting {file_path} (Last modified: {datetime.fromtimestamp(file_mtime)})")
                os.remove(file_path)

# Example usage:
directory = '/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/job_output'
delete_old_files(directory)