import time
import subprocess
from collections import defaultdict
import json
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="outputs", help='output result directory')
    parser.add_argument('--freq', type=int, default=1, help='freq of cpu hz data')
    args = parser.parse_args()
        
    # Define the file name to store the logs
    log_file = os.path.join(args.output_dir, "cpu_freq.log")

    # Define the dictionary to store the CPU frequency data
    cpu_freq = defaultdict(list)

    try:
        while True:
            # Execute the grep command
            grep_cmd = subprocess.Popen(["grep", "MHz", "/proc/cpuinfo"], stdout=subprocess.PIPE)

            # Parse the output and store the Hz information in the dictionary
            core_id = 0
            for line in grep_cmd.stdout:
                freq = float(line.split()[3])
                cpu_freq[f'core_{core_id}'].append(freq)
                core_id += 1

            # Wait for 1 second before logging again
            time.sleep(args.freq)
    except KeyboardInterrupt:
        # Write the dictionary to the log file
        with open(log_file, "w") as f:
            f.write(json.dumps(dict(cpu_freq), indent=4))
        exit()