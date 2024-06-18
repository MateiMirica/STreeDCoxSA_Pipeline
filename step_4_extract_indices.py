import numpy as np
import os
import shutil
from utils import files_in_directory, parse_line
from utils import ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY, BINARY_DIRECTORY, INDICES_DIRECTORY

SEED = 4136121025
np.random.seed(SEED)

K = 5

def main():
    # Create necessary directories
    output_parent_directory = "/".join(INDICES_DIRECTORY.split("/")[:-1])
    if not os.path.exists(output_parent_directory):
        os.mkdir(output_parent_directory)
    if not os.path.exists(INDICES_DIRECTORY):
        os.mkdir(INDICES_DIRECTORY)

    # Empty each train/test-directory
    for directory in [INDICES_DIRECTORY]:
        for section in ["train", "test"]:
            path = f"{directory}/{section}"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)

    for filename in [j[:-4] for j in files_in_directory(ORIGINAL_DIRECTORY) if not j.startswith("generated")]:
        # Read instances from file
        f = open(f"{ORIGINAL_DIRECTORY}/{filename}.txt")
        init_lines = f.read().strip().split("\n")
        f.close()



        # Extract train/test-files indices:
        for i in range(K):
            dict = {}
            for j, inst in enumerate(init_lines[1:]):
                if inst not in dict:
                    dict[inst] = []
                dict[inst].append(j)

            for t in ["train", "test"]:
                # Read lines from file
                f = open(f"{ORIGINAL_DIRECTORY}/{t}/{filename}_partition_{i}.txt")
                lines = f.read().strip().split("\n")
                f.close()
                data_lines = lines[1:]
                instances = []
                for line in data_lines:
                    instances.append(str(dict[line].pop()))
                f = open(f"{INDICES_DIRECTORY}/{t}/{filename}_partition_{i}.txt", "w")
                f.write("\n".join(instances))
                f.close()

        print(f"\033[35mSplit \033[1m{filename}\033[0m")

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()
