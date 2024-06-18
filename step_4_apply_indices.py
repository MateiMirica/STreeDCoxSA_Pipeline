import numpy as np
import os
import shutil
from utils import files_in_directory, parse_line
from utils import ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY, BINARY_DIRECTORY, INDICES_DIRECTORY

SEED = 4136121025
np.random.seed(SEED)

K = 5

def main():
    for directory in [ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY, BINARY_DIRECTORY]:
        for section in ["train", "test"]:
            path = f"{directory}/{section}"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)

    for filename in [j[:-4] for j in files_in_directory(ORIGINAL_DIRECTORY) if not j.startswith("generated")]:
        for directory in [ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY, BINARY_DIRECTORY]:
            # Read instances from file
            f = open(f"{directory}/{filename}.txt")
            init_lines = f.read().strip().split("\n")
            f.close()
            info_line = init_lines[0] + "\n"
            init_lines = init_lines[1:]
            # Generate train/test-files files:
            for i in range(K):
                for t in ["train", "test"]:
                    # Read lines from file
                    f = open(f"{INDICES_DIRECTORY}/{t}/{filename}_partition_{i}.txt")
                    lines = f.read().strip().split("\n")
                    f.close()
                    data_lines = lines
                    instances = []
                    for line in data_lines:
                        instances.append(init_lines[int(line)])
                    f = open(f"{directory}/{t}/{filename}_partition_{i}.txt", "w")
                    f.write(info_line)
                    f.write("\n".join(instances))
                    f.close()

        print(f"\033[35mSplit \033[1m{filename}\033[0m")

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()
