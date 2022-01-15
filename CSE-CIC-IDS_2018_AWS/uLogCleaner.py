import os

def line_cleaner(line):
    cleaned_line = "".join(c if c.isalpha() or c.isdigit() else " " for c in line.lower())
    return " ".join(cleaned_line.split())


def concatenate_logs():
    # get the current working directory
    working_dir = os.getcwd()

    # loop through all directories in the working directory
    for dir_name in os.listdir(working_dir):
        # check if the directory name ends with "_Ubuntu" and is a directory
        if dir_name.endswith("_Ubuntu") and os.path.isdir(dir_name):
            print(f"Processing directory: {dir_name}")
            # concatenate all log files in the directory
            log_data = ""
            for file_name in os.listdir(dir_name):
                try:
                    with open(os.path.join(dir_name, file_name), "r", encoding="utf-8") as f:
                        for line in f:
                            line = " ".join(line.split()[4:])
                            cleaned_line = line_cleaner(line)
                            log_data += cleaned_line + "\n"
                except UnicodeDecodeError:
                    print(f"Error reading file: {os.path.join(dir_name, file_name)}")
            # write the concatenated data to a new file in the proper directory
            if log_data:
                new_file_name = dir_name + ".concatenated.log"
                new_dir_name = os.path.join("preprocessed_logs", dir_name)
                if not os.path.exists(new_dir_name):
                    os.makedirs(new_dir_name)
                with open(os.path.join(new_dir_name, new_file_name), "w", encoding="utf-8") as f:
                    f.write(log_data)
                print(f"Wrote file: {os.path.join(new_dir_name, new_file_name)}")

def main():
    concatenate_logs()

if __name__ == "__main__":
    main()
