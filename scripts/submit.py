import os
import subprocess

def submit_and_delete_sh_files(start_directory):
    """
    Search the file tree from the given directory, submit all .sh files using sbatch, and delete them.

    Parameters:
        start_directory (str): The directory to start searching from.
    """
    # Walk through the directory tree
    for root, _, files in os.walk(start_directory):
        for file in files:
            if file.endswith(".sh"):
                file_path = os.path.join(root, file)
                try:
                    # Submit the .sh file using sbatch
                    print(f"Submitting: {file_path}")
                    subprocess.run(["sbatch", file_path], check=True)

                    # Delete the file after submission
                    print(f"Deleting: {file_path}")
                    os.remove(file_path)
                except subprocess.CalledProcessError as e:
                    print(f"Error submitting {file_path}: {e}")
                except Exception as e:
                    print(f"Error handling {file_path}: {e}")

if __name__ == "__main__":
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Submit and delete .sh files using SLURM.")
    parser.add_argument("start_directory", type=str, help="The directory to start searching from.")

    # Parse arguments
    args = parser.parse_args()

    # Call the function
    submit_and_delete_sh_files(args.start_directory)
