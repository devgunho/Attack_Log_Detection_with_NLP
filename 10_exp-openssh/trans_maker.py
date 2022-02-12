import os
import random


def get_openssh_rand_word(line):
    """Get a random word from a line of text."""
    words = line.split()
    # remove timestamp
    words = words[8:]
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
    file = open("../utils/cleaned_datasets/OpenSSH_2k.log_clean.txt", "r")
    lines = file.readlines()

    trans_file_dir = "data/trans"
    if not os.path.exists(trans_file_dir):
        os.makedirs(trans_file_dir)

    first_line = lines[0].strip()
    print(first_line)

    for cnt in range(1, 21):
        result_file = open(
            os.path.join(trans_file_dir, f"concordance_" + str(cnt) + "_clean.lam"),
            "w",
        )

        for line in lines:
            target_line = line.strip()

            ramdom_word = get_openssh_rand_word(target_line)
            result_file.write(
                f"exists x1.(_{ramdom_word[0]}(x1) & exists x2.(_{ramdom_word[1]}(x2) & (x1 = x2)))"
                + "\n"
            )

        result_file.close()
