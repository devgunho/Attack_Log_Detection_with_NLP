import os
import random

TARGET_FILE = "./preprocessed_logs/2018-02-03_Ubuntu/2018-02-03_Ubuntu.concatenated.log"

def get_rand_word(line):
    line = line.strip()
    line_list = line.split()
    random_tok = random.sample(line_list, k=2)

    loop = 0
    while (
        str(random_tok[0]).isnumeric() == True
        or str(random_tok[1]).isnumeric() == True
        or (len(random_tok[0]) + len(random_tok[1]) <= 2)
        or len(random_tok[0]) == 1
        or len(random_tok[1]) == 1
    ):
        random_tok = random.sample(line_list, k=2)
        loop += 1
        if loop == 10:
            break
        print(random_tok)

    return random_tok

file_cnt = 1
while True:
    file = open(TARGET_FILE, "r")
    lines = file.readlines()

    result_file_name = f"concordance_" + str(file_cnt) + "_clean.lam"
    if not os.path.exists("./data/trans/"):
        os.makedirs("./data/trans/")
    result_path = os.path.join("./data/trans/", result_file_name)
    result_file = open(result_path, "w")

    for line in lines:
        target_line = line.strip()
        random_tok = get_rand_word(target_line)
        result_file.write(
            f"exists x1.(_{random_tok[0]}(x1) & exists x2.(_{random_tok[1]}(x2) & (x1 = x2)))"
            + "\n"
        )

    result_file.close()
    file_cnt += 1
    if file_cnt == 21:
        break