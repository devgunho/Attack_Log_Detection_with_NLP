# Make clean dataset for anomaly detection
# Expect error log lines as input
import os

ORIGIN_FILE_PATH = "../02_loghub-datasets/Apache/Apache_2k.log"
TIMESTAMP_LENGTH = 7


def line_cleaner(line):
    concat_char = ""

    for char in line:
        try:
            if char.isdigit():
                concat_char += char
            elif char.islower():
                concat_char += char
            elif char.isupper():
                char = char.lower()
                concat_char += char
            elif char == " ":
                if concat_char[-1] == " ":
                    pass
                else:
                    concat_char += " "
                pass
            else:
                if len(concat_char) == 0:
                    pass
                elif concat_char[-1] == " ":
                    pass
                else:
                    concat_char += " "
                pass

        except Exception as e:
            print(e)
            raise e
    return concat_char


def make_clean_dataset(origin_file_path):
    # Make dir for clean dataset
    dst_dir_path = "./data/clean"
    anomaly_dst_dir_path = "./data/anomaly_clean"
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    if not os.path.exists(anomaly_dst_dir_path):
        os.makedirs(anomaly_dst_dir_path)

    # Read origin file
    with open(origin_file_path, "r") as ori_f:
        lines = ori_f.readlines()
        print(f"Total lines: {len(lines)}")

        # Write clean dataset
        for file_cnt in range(1, 21):
            dst = os.path.join(dst_dir_path, f"./concordance_{file_cnt}_clean.txt")
            anomaly_dst = os.path.join(
                anomaly_dst_dir_path, f"./concordance_{file_cnt}_clean.txt"
            )
            print(f"Writing {dst}")

            with open(dst, "w") as dst_f:
                with open(anomaly_dst, "w") as anomaly_dst_f:
                    for line in lines:
                        print("[Original line]: ", line.strip())
                        cleaned_line = line_cleaner(line.strip())
                        print("[Cleaned line]: ", cleaned_line)
                        cleaned_line = cleaned_line.split(" ")[TIMESTAMP_LENGTH:]
                        print("[Exclude timestamp]: ", " ".join(cleaned_line))

                        # if error log, write to anomaly dataset
                        if cleaned_line[0] != "error":
                            dst_f.write(" ".join(cleaned_line) + "\n")
                        else:
                            anomaly_dst_f.write(" ".join(cleaned_line) + "\n")


if __name__ == "__main__":
    make_clean_dataset(ORIGIN_FILE_PATH)
