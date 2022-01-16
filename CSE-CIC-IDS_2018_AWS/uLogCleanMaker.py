from shutil import copyfile
import os

file_cnt = 0
TARGET_FILE = "./preprocessed_logs/2018-02-03_Ubuntu/2018-02-03_Ubuntu.concatenated.log"

while True:
    src = TARGET_FILE
    dir_path = f"./data/clean/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_cnt+=1
    file_name = f"concordance_{file_cnt}.txt"
    dst = f"./data/clean/concordance_{file_cnt}_clean.txt"
    copyfile(src,dst)
    if file_cnt == 20:
        break;