import os
import time
from datetime import datetime, timedelta

def copy_lines(input_file, output_file):
    try:
        # Open the input file in read mode and output file in write mode
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            # Iterate over each line in the input file
            for line in infile:
                # Check if the line ends with "Start training!"
                if line.strip().endswith("Starting training!"):
                    # Write the line to the output file
                    outfile.write(line)
                    print(line)

        print(f"Lines ending with 'Start training!' have been copied to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = "/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/job_output/TRANSITv16f_group-13146811-TRANSITv16f_group_4294967294.out"  # Replace with your input file path
output_file = "lines.txt"  # Replace with your output file path
copy_lines(input_file, output_file)