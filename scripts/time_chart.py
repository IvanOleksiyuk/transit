import matplotlib.pyplot as plt
import datetime
import argparse

def read_task_times(file_path):
    task_durations = {}
    with open(file_path, "r") as file:
        for line in file:
            if ":" in line and "Start time" not in line:
                parts = line.split(":")
                if len(parts) > 2:
                    task_name = parts[0].strip()
                    time_str = line.split(task_name + ": ")[1].strip()
                    try:
                        time_delta = datetime.timedelta(
                            hours=int(time_str.split(":")[0]),
                            minutes=int(time_str.split(":")[1]),
                            seconds=float(time_str.split(":")[2])
                        )
                        task_durations[task_name] = time_delta.total_seconds()
                    except Exception as e:
                        print(f"Error parsing line: {line}. Error: {e}")
    return task_durations

def plot_pie_chart(task_durations, save_path=None):
    labels = list(task_durations.keys())
    times = list(task_durations.values())
    total_time = sum(times)

    plt.figure(figsize=(10, 8))
    plt.pie(times, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f"Time Spent on Each Task (Total: {str(datetime.timedelta(seconds=int(total_time)))})")
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def main(file_path, save_path=None):
    task_durations = read_task_times(file_path)
    if task_durations:
        plot_pie_chart(task_durations, save_path=save_path)
    else:
        print("No valid task times found in the file.")

def run():
    parser = argparse.ArgumentParser(description="Plot time spent on tasks from a file.")
    parser.add_argument("file", help="Path to the file containing task times.")
    parser.add_argument("--save", help="Path to save the generated figure.", default=None)
    args = parser.parse_args()

    main(args.file, args.save)

if __name__ == "__main__":
    run()
    
#/opt/conda/bin/python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/time_chart.py /home/users/o/oleksiyu/WORK/hyperproject/workspaces/TEST_jobs/TRANSITv0v1_LHCO/summary/runtime.txt --save time_plot.png