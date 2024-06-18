import numpy as np
import os
import shutil
from utils import files_in_directory, parse_line
from utils import ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY, BINARY_DIRECTORY

SEED = 4136121025
np.random.seed(SEED)

K = 5

def main():
    # Empty each train/test-directory
    for directory in [ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY, BINARY_DIRECTORY]:
        for section in ["train", "test"]:
            path = f"{directory}/{section}"
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)

    for filename in [j[:-4] for j in files_in_directory(ORIGINAL_DIRECTORY) if not j.startswith("generated")]:
        # Read instances from file
        f = open(f"{ORIGINAL_DIRECTORY}/{filename}.txt")
        lines = f.read().strip().split("\n")
        f.close()

        instances = []
        for line in lines[1:]:
            inst = parse_line(line)
            instances.append(inst)
        f.close()

        # Group instances based on observation status
        indices_per_event = [[], []]
        for i in range(len(instances)):
            indices_per_event[instances[i][1]].append(i)

        # Take `100c` percent of each group, make `K` partitions
        partitions = [set() for _ in range(K)]
        for event in range(2):
            curr_indices = indices_per_event[event]
            np.random.shuffle(curr_indices)
            curr_partitions = [{*curr_indices[j * len(curr_indices) // K:(j + 1) * len(curr_indices) // K]} for j in range(K)]
            if event == 1:
                curr_partitions = curr_partitions[::-1]
            for i in range(K):
                partitions[i] |= curr_partitions[i]

        # Create train/test-files for each partition
        for i, partition in enumerate(partitions):
            for directory in [ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY, BINARY_DIRECTORY]:
                # Read lines from file
                f = open(f"{directory}/{filename}.txt")
                lines = f.read().strip().split("\n")
                f.close()

                info_line = lines[0] + "\n"
                data_lines = lines[1:]

                # Divide lines into train- and test-group
                train_lines, test_lines = [], []
                for j, line in enumerate(data_lines):
                    if j in partition:
                        test_lines.append(line)
                    else:
                        train_lines.append(line)

                # Write the files
                for section, lines in [("train", train_lines), ("test", test_lines)]:
                    path = f"{directory}/{section}/{filename}_partition_{i}.txt"

                    f = open(path, "w")
                    f.write(info_line)
                    f.write("\n".join(lines))
                    f.close()

        print(f"\033[35mSplit \033[1m{filename}\033[0m")

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()
