import os

file_cnt = 1
lines_per_file = 10000
target_file = "./preprocessed_logs/concatenated_all_logs.log"
num_files = 20

with open(target_file, "r") as input_file:
    while file_cnt <= num_files:
        output_dir = "./data/clean/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"concordance_{file_cnt}_clean.txt")

        with open(output_file, "w") as output:
            for _ in range(lines_per_file):
                line = input_file.readline()
                if not line:
                    input_file.seek(0)  # Reset the file pointer to the beginning
                    line = input_file.readline()  # Read the first line again
                output.write(line)

        if os.path.getsize(output_file) == 0:
            os.remove(output_file)
            break

        file_cnt += 1
        print(f"Created {output_file}")
