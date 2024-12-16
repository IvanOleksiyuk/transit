import os
import glob

def find_files_with_name(root_dir, filename):
    # Use glob to find all files matching the filename pattern
    search_pattern = os.path.join(root_dir, '**', filename)
    matching_files = glob.glob(search_pattern, recursive=True)
    return matching_files