import os
import shutil
import fnmatch

def delete_npy_files_in_cwola(start_directory):
    # Walk through the directory tree
    for root, dirs, files in os.walk(start_directory):
        # Check if the current folder is named "cwola"
        if os.path.basename(root) == "dope_3000":
            # Loop through files in the current "cwola" folder
            for file in files:
                if file.endswith(".npy"):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)  # Delete the .npy file
                        print(f"Deleted {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)  # Delete the .npy file
                        print(f"Deleted {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

def delete_by_patterns(directory_path, patterns, dry_run=False):
    """
    Deletes files and folders based on the patterns provided, with an option for a dry run.
    Also computes the total disk space that would be freed.

    Patterns can specify both directories and file names, like 'logs/*.txt' to delete only .txt files in 'logs' directories.

    Args:
        directory_path (str): The root directory where the search will start.
        patterns (list): List of patterns combining directories and file names, e.g., ['logs/*.txt', 'wandb/*', '*.log'].
        dry_run (bool): If True, only list the files and folders that would be deleted without actually deleting them.

    Returns:
        int: The total disk space that would be freed (in bytes).
    """
    total_freed_space = 0  # To keep track of the total space freed

    for root, dirs, files in os.walk(directory_path, topdown=False):  # topdown=False ensures we delete bottom-up
        # Delete files that match any pattern in the given directory
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Check if any of the provided patterns match the file path
            for pattern in patterns:
                dir_pattern, file_pattern = os.path.split(pattern)

                # Match file patterns only in the specific directories provided in the pattern
                if fnmatch.fnmatch(os.path.basename(root), dir_pattern) or dir_pattern == '':
                    if fnmatch.fnmatch(file_name, file_pattern):
                        file_size = os.path.getsize(file_path)  # Get file size for space calculation
                        total_freed_space += file_size
                        if dry_run:
                            print(f"Dry run - would delete file: {file_path} (Size: {file_size} bytes)")
                        else:
                            try:
                                os.remove(file_path)
                                print(f"Deleted file: {file_path} (Freed: {file_size} bytes)")
                            except Exception as e:
                                print(f"Error deleting file {file_path}: {e}")

        # Delete folders that match the folder pattern
        for dir_name in dirs:
            for pattern in patterns:
                dir_pattern, file_pattern = os.path.split(pattern)

                # If the pattern is for a folder and does not specify a file
                if fnmatch.fnmatch(dir_name, dir_pattern) and file_pattern == '':
                    folder_path = os.path.join(root, dir_name)
                    folder_size = get_folder_size(folder_path)  # Calculate folder size
                    total_freed_space += folder_size
                    if dry_run:
                        print(f"Dry run - would delete folder: {folder_path} (Size: {folder_size} bytes)")
                    else:
                        try:
                            shutil.rmtree(folder_path)  # Deletes the entire directory and its contents
                            print(f"Deleted folder: {folder_path} (Freed: {folder_size} bytes)")
                        except Exception as e:
                            print(f"Error deleting folder {folder_path}: {e}")

    return total_freed_space

def find_matching_paths(root_dir, patterns, delete=False):
    """
    Finds all files and directories that match one of the provided patterns.

    :param root_dir: The root directory to start searching from.
    :param patterns: A list of patterns to match against the file or folder names.
    :return: A list of paths that match the patterns.
    """
    matching_paths = []
    
    # Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check directories
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            for pattern in patterns:
                if fnmatch.fnmatch(full_path, os.path.join(root_dir, pattern)):
                    print(full_path)
                    if delete:
                        delete_path(full_path)
                    matching_paths.append(full_path)
        
        # Check files
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            for pattern in patterns:
                if fnmatch.fnmatch(full_path, os.path.join(root_dir, pattern)):
                    print(full_path)
                    if delete:
                        delete_path(full_path)
                    matching_paths.append(full_path)
    
    return matching_paths

def get_folder_size(folder_path):
    """
    Calculate the total size of all files in a folder and its subdirectories.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        int: The total size of the folder in bytes.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def delete_path(path):
    """
    Deletes the specified file or directory.

    :param path: The path to the file or directory to be deleted.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            # It's a file, delete it
            os.remove(path)
            print(f"File deleted: {path}")
        elif os.path.isdir(path):
            # It's a directory, delete it and all its contents
            shutil.rmtree(path)
            print(f"Directory deleted: {path}")
    else:
        print(f"Path does not exist: {path}")

if __name__ == "__main__":
    # Replace this with the starting directory path you want to search in
    start_directory = "/home/users/o/oleksiyu/WORK/hyperproject/workspaces/"
    #delete_npy_files_in_cwola(start_directory)
    #total_freed = delete_by_patterns(start_directory, ["input/**.png"], dry_run=True)
    #print(f"Total disk space freed: {total_freed/1024/1024} MB")
    
    #find_matching_paths(start_directory, ["**/wandb", "**/wandb"], delete=True)
    #find_matching_paths(start_directory, ["**/plots/input", "**/plots/inputs"], delete=True)
    #find_matching_paths(start_directory, ["**/cwola", "**/cwola_latent"], delete=True)
    find_matching_paths(start_directory, ["**/cwola", "**/cwola_latent"], delete=True)