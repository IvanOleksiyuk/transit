import os
import subprocess

def submit_slurm_jobs(directory, filename_prefix):
    # Change the working directory to home
    os.chdir(os.path.expanduser("~"))
    
    for filename in os.listdir(directory):
        if filename.startswith(filename_prefix):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                print(f"Submitting job: {filepath}")
                result = subprocess.run(['sbatch', filepath], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"Job submitted successfully: {result.stdout}")
                else:
                    print(f"Failed to submit job: {result.stderr}")

if __name__ == "__main__":
    # directories = ["/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/experiments_2d_gauss_nom_EXPEVAL",
    #                "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/experiments_4d_gauss_usem_EXPEVAL",
    #                "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/experiments_8d_gauss_usem_EXPEVAL",
    #                "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/gauss_corr_4_gap_transit_usem_EXPEVAL",
    #                "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/powers_5_gap_transit_usem_EXPEVAL",
    #                "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/swiss_roll_gap_transit_usem_EXPEVAL",]  # Change this to your specific folder
    directories = ["/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/llv_idealised"]
    
    filename_prefix = "TRANSIT"
    for directory in directories:
        submit_slurm_jobs(directory, filename_prefix)