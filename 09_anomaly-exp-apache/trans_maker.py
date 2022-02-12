import os
import random


def get_rand_word(line):
    """Get a random word from a line of text."""
    words = line.split()
    random_two_words = random.sample(words, 2)

    max_iteration = 10
    while (
        str(random_two_words[0]).isnumeric() == True
        or str(random_two_words[1]).isnumeric() == True
        or (len(random_two_words[0]) + len(random_two_words[1]) <= 2)
        or len(random_two_words[0]) == 1
        or len(random_two_words[1]) == 1
    ):
        random_two_words = random.sample(words, 2)
        max_iteration -= 1
        if max_iteration == 0:
            break
        print("words", words)
        print("random_two_words", random_two_words)
        print("max_iteration", max_iteration)

    return random_two_words


if __name__ == "__main__":
    normal_file = open("./data/clean/concordance_1_clean.txt", "r")
    normal_lines = normal_file.readlines()
    normal_trans_file_dir = "./data/trans"
    if not os.path.exists(normal_trans_file_dir):
        os.makedirs(normal_trans_file_dir)

    anomaly_file = open("./data/anomaly_clean/concordance_1_clean.txt", "r")
    anomaly_lines = anomaly_file.readlines()
    anomaly_trans_file_dir = "./data/anomaly_trans"
    if not os.path.exists(anomaly_trans_file_dir):
        os.makedirs(anomaly_trans_file_dir)

    first_line = normal_lines[0].strip()
    print("[NORMAL] Check first line:", first_line)

    first_line = anomaly_lines[0].strip()
    print("[ANOMALY] Check first line:", first_line)

    for cnt in range(1, 21):
        # normal
        normal_result_file = open(
            os.path.join(
                normal_trans_file_dir, f"concordance_" + str(cnt) + "_clean.lam"
            ),
            "w",
        )

        for n_line in normal_lines:
            target_line = n_line.strip()
            random_word = get_rand_word(target_line)
            normal_result_file.write(
                f"exists x1.(_{random_word[0]}(x1) & exists x2.(_{random_word[1]}(x2) & (x1 = x2)))"
                + "\n"
            )

        normal_result_file.close()

        # anomaly
        anomaly_result_file = open(
            os.path.join(
                anomaly_trans_file_dir, f"concordance_" + str(cnt) + "_clean.lam"
            ),
            "w",
        )

        for a_line in anomaly_lines:
            target_line = a_line.strip()
            random_word = get_rand_word(target_line)
            anomaly_result_file.write(
                f"exists x1.(_{random_word[0]}(x1) & exists x2.(_{random_word[1]}(x2) & (x1 = x2)))"
                + "\n"
            )

        anomaly_result_file.close()
