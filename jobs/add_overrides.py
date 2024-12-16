import os
import re


def make_change_time(lines, time_str):
    modified_lines = []
    if time_str is not None:
        for line in lines:
            if "#SBATCH --time=" in line:
                modified_lines.append(time_str)
            else:
                modified_lines.append(line)
    return modified_lines

def modify_slurm_jobs(source_dir, target_dir, line_part_to_find, addition_to_line, change_time=None):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".sh"):  # Assuming SLURM job files have a .sh extension
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)

            with open(source_file, 'r') as f:
                lines = f.readlines()

            modified_lines = []
            for line in lines:
                if line_part_to_find in line:
                    # Check if the line ends with a quote
                    if line.strip().endswith('"'):
                        match = re.search(r'--config-name [^\s"]+', line)
                        if match:
                            start_pos = match.end()
                            modified_line = (
                                line[:start_pos] +
                                f' {addition_to_line}' +
                                line[start_pos:]
                            )
                        else:
                            modified_line = line
                    # Check if the line ends with a backslash
                    elif line.strip().endswith('\\'):
                        match = re.search(r'--config-name [^\s\\]+', line)
                        if match:
                            start_pos = match.end()
                            modified_line = (
                                line[:start_pos] +
                                f' {addition_to_line}' +
                                line[start_pos:]
                            )
                        else:
                            modified_line = line
                    else:
                        modified_line = line  # No match found; keep the line unchanged
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)
            
            if change_time is not None:
                modified_lines = make_change_time(modified_lines, change_time)
            
            with open(target_file, 'w') as f:
                f.writelines(modified_lines)

if __name__ == "__main__":
    source_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/experiments_2d_gauss_usem"  # Change this to your source directory
    line_part_to_find = "python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run.py"
    
    target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/experiments_2d_gauss_nom"  # Change this to your target directory
    addition_to_line = "data=gauss_corr_2_transit_nom general.subfolder=gauss_corr_2_transit_nom/"
    modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line)
    
    target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/experiments_4d_gauss_usem"  # Change this to your target directory
    addition_to_line = "data=gauss_corr_4_transit_usem general.subfolder=gauss_corr_4_transit_usem/"
    modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line)
    
    target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/experiments_8d_gauss_usem"  # Change this to your target directory
    addition_to_line = "data=gauss_corr_8_transit_usem general.subfolder=gauss_corr_8_transit_usem/"
    modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line, change_time="#SBATCH --time=2:00:00\n")
    
    target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/experiments_4d_gausssph_usem"  # Change this to your target directory
    addition_to_line = "data=gauss_sph_4_transit_usem general.subfolder=gauss_sph_4_transit_usem/"
    modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line)

    target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/gauss_corr_4_gap_transit_usem"  # Change this to your target directory
    addition_to_line = "data=gauss_corr_4_gap_transit_usem general.subfolder=gauss_corr_4_gap_transit_usem/"
    modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line)

    target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/powers_5_gap_transit_usem"  # Change this to your target directory
    addition_to_line = "data=powers_5_gap_transit_usem general.subfolder=powers_5_gap_transit_usem/"
    modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line)

    target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/swiss_roll_gap_transit_usem"  # Change this to your target directory
    addition_to_line = "data=swiss_roll_gap_transit_usem general.subfolder=swiss_roll_gap_transit_usem/"
    modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line)

    target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/gauss_corr_4_gap_transit_usem_nogap_train"  # Change this to your target directory
    addition_to_line = "data=gauss_corr_4_gap_transit_usem_nogap_train general.subfolder=gauss_corr_4_gap_transit_usem_nogap_train/"
    modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line)

    target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/powers_5_gap_transit_usem_nogap_train"  # Change this to your target directory
    addition_to_line = "data=powers_5_gap_transit_usem_nogap_train general.subfolder=powers_5_gap_transit_usem_nogap_train/"
    modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line)

    target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/swiss_roll_gap_transit_usem_nogap_train"  # Change this to your target directory
    addition_to_line = "data=swiss_roll_gap_transit_usem_nogap_train general.subfolder=swiss_roll_gap_transit_usem_nogap_train/"
    modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line)
    # Rerun expport and eval 
    # target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/experiments_2d_gauss_nom_EXPEVAL"  # Change this to your target directory
    # addition_to_line = "data=gauss_corr_2_transit_nom general.subfolder=gauss_corr_2_transit_nom/ do_train_template=0"
    # modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line, change_time="#SBATCH --time=0:20:00\n")
    
    # target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/experiments_2d_gauss_usem_EXPEVAL"  # Change this to your target directory
    # addition_to_line = "do_train_template=0"
    # modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line, change_time="#SBATCH --time=0:20:00\n")
    
    # target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/experiments_4d_gauss_usem_EXPEVAL"  # Change this to your target directory
    # addition_to_line = "data=gauss_corr_4_transit_usem general.subfolder=gauss_corr_4_transit_usem/ do_train_template=0"
    # modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line, change_time="#SBATCH --time=0:20:00\n")
    
    # target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/experiments_8d_gauss_usem_EXPEVAL"  # Change this to your target directory
    # addition_to_line = "data=gauss_corr_8_transit_usem general.subfolder=gauss_corr_8_transit_usem/ do_train_template=0"
    # modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line, change_time="#SBATCH --time=0:20:00\n")

    # target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/gauss_corr_4_gap_transit_usem_EXPEVAL"  # Change this to your target directory
    # addition_to_line = "data=gauss_corr_4_gap_transit_usem general.subfolder=gauss_corr_4_gap_transit_usem/ do_train_template=0"
    # modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line, change_time="#SBATCH --time=0:20:00\n")

    # target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/powers_5_gap_transit_usem_EXPEVAL"  # Change this to your target directory
    # addition_to_line = "data=powers_5_gap_transit_usem general.subfolder=powers_5_gap_transit_usem/ do_train_template=0"
    # modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line, change_time="#SBATCH --time=0:20:00\n")

    # target_directory = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/IGNORE_overrides/swiss_roll_gap_transit_usem_EXPEVAL"  # Change this to your target directory
    # addition_to_line = "data=swiss_roll_gap_transit_usem general.subfolder=swiss_roll_gap_transit_usem/ do_train_template=0"
    # modify_slurm_jobs(source_directory, target_directory, line_part_to_find, addition_to_line, change_time="#SBATCH --time=0:20:00\n")
    