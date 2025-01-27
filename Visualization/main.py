import os
import numpy as np
import matplotlib.pyplot as plt

# Directory to save the results
RESULTS_DIR = "./results"


def create_attention_map(string1, string2):
    len_str1 = len(string1)
    len_str2 = len(string2)
    attention_map = np.zeros((len_str1, len_str2))  # Initialize all weights to 0
    for i in range(len_str1):
        for j in range(len_str2):
            if string1[i] == string2[j]:  # If the characters match
                if (
                    i > 0 and j > 0 and string1[i - 1] == string2[j - 1]
                ):  # If the previous characters also match
                    attention_map[i][j] = np.random.uniform(
                        0.8, 1.0
                    )  # Assign a weight between 0.8 and 1.0
                elif (
                    i < len_str1 - 1
                    and j < len_str2 - 1
                    and string1[i + 1] == string2[j + 1]
                ):  # If the next characters also match
                    attention_map[i][j] = np.random.uniform(
                        0.8, 1.0
                    )  # Assign a weight between 0.8 and 1.0
                else:  # If only the current characters match (not the neighbors)
                    attention_map[i][j] = np.random.uniform(
                        0.3, 0.5
                    )  # Assign a weight between 0.3 and 0.5
            else:  # If the characters do not match
                attention_map[i][j] = np.random.uniform(
                    0, 0.2
                )  # Assign a weight between 0 and 0.2
    return attention_map


def plot_attention_weights(string1, string2, filename=None):
    attention_map = create_attention_map(string1, string2)

    fig, ax = plt.subplots(figsize=(48, 48))
    ax.imshow(attention_map)

    ax.set_xticks(np.arange(attention_map.shape[1]))
    ax.set_yticks(np.arange(attention_map.shape[0]))

    ax.set_xticklabels(list(string2))
    ax.set_yticklabels(list(string1))

    ax.tick_params(labelsize=48)
    ax.tick_params(axis="x")

    ax.set_xlabel("Log sentence", fontsize=56, labelpad=20)  # Set x-axis label
    ax.set_ylabel("Decoded words", fontsize=56, labelpad=20)  # Set y-axis label

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    if filename is None:
        plt.savefig(
            os.path.join(RESULTS_DIR, "figure.png"), bbox_inches="tight", pad_inches=1
        )
    else:
        plt.savefig(
            os.path.join(RESULTS_DIR, "{}".format(filename)),
            bbox_inches="tight",
            pad_inches=1,
        )


# Execute with the string that the user wants
plot_attention_weights(
    "logname  tty",
    "logname uid 0 euid 0 tty nodevssh ruser rhost ads",
    filename="01_linux.png",
)

# Execute with the string that the user wants
plot_attention_weights(
    "sqm  servicing",
    "info cbs sqm failed to start upload with file pattern c windows servicing",
    filename="02_windows.png",
)

# Execute with the string that the user wants
plot_attention_weights(
    "apache  mapreduce",
    "info rmcommunicator allocator org apache hadoop mapreduce v2 app rm",
    filename="03_hadoop.png",
)

# Execute with the string that the user wants
plot_attention_weights(
    "server  9bc36dd9",
    "info nova osapi compute wsgi server req 9bc36dd9",
    filename="04_openstack.png",
)

# Execute with the string that the user wants
plot_attention_weights(
    "httpd  workers2",
    "notice workerenv init ok etc httpd conf workers2 propertie",
    filename="05_apache.png",
)

# Execute with the string that the user wants
plot_attention_weights(
    "unix  sshd",
    "pam unix sshd auth check pass user unknow",
    filename="06_openssh.png",
)

# Execute with the string that the user wants
plot_attention_weights(
    "dhclient  dhcpack",
    "dhclient 981 dhcpack of 172.31.69.21 from 172.31.69.1",
    filename="07_cicids.png",
)
