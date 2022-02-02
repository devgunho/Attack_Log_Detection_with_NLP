import os
import random

def get_rand_word(line):
    line = line.strip()
    line_list = line.split()

    loop = 0
    random_tok = random.sample(line_list, k=2)
    while (
        any(tok.isnumeric() for tok in random_tok)
        or sum(len(tok) for tok in random_tok) <= 2
        or any(len(tok) == 1 for tok in random_tok)
    ):
        random_tok = random.sample(line_list, k=2)
        loop += 1
        if loop == 10:
            break
        print(random_tok)

    return random_tok


file_cnt = 1
lines_per_file = 10000
target_file = "./preprocessed_logs/concatenated_all_logs.log"
num_files = 20

with open(target_file, "r") as input_file:
    while file_cnt <= num_files:
        output_dir = "./data/trans/"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"concordance_{file_cnt}_clean.lam")

        with open(output_file, "w") as output:
            for _ in range(lines_per_file):
                line = input_file.readline()
                if not line:
                    input_file.seek(0)  # Reset the file pointer to the beginning
                    line = input_file.readline()  # Read the first line again
                random_word = get_rand_word(line)
                result = f"exists x1.(_{random_word[0]}(x1) & exists x2.(_{random_word[1]}(x2) & (x1 = x2)))"
                output.write(result + "\n")

        if os.path.getsize(output_file) == 0:
            os.remove(output_file)
            break

        file_cnt += 1
        print(f"Created {output_file}")
