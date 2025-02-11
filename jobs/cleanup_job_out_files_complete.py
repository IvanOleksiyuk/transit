import os

# Define the folder path
folder_path = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/job_output"  # Change this to your target folder

# Define the target string
target_string = "<<<END FULL RUN>>>"

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Ensure it's a file (not a directory)
    if os.path.isfile(file_path):
        try:
            # Read the file and check for the target string
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                if target_string in file.read():
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")