import numpy as np
from math import exp, pow
from sksurv.metrics import integrated_brier_score
from utils import fill_tree, parse_tree, read_dataset, Tree, parse_cox_tree, fill_cox_tree
from utils import DIRECTORY, ORIGINAL_DIRECTORY, NUMERIC_DIRECTORY

# Calculates the Harrell's C-index (concordance) for a certain tree
#
# root          The tree to evaluate the instances with
# instances     The instances to calculate the concordance with
def calculate_concordance(root, instances, method="leblanc"):
    # Group the instances together based on their theta
    buckets = {}
    
    if True or method == "leblanc":
        for inst in instances:
            theta, _, _ = root.classify(inst)
            if theta not in buckets:
                buckets[theta] = [[], [], []] # [censored, death, all]
            buckets[theta][inst.event].append(inst)
            buckets[theta][2].append(inst)
        buckets = [buckets[key] for key in sorted(buckets.keys())]
        for i in range(len(buckets)):
            for j in range(3):
                buckets[i][j] = sorted(buckets[i][j], key=lambda x: x.time)

        # Apply binary search to compute the amount of discordance and concordant pairs efficiently
        cc = tr = dc = 0
        for i in range(len(buckets)):
            for j in range(len(buckets)):
                for inst_i in buckets[i][2]:
                    a, b = 0, len(buckets[j][1])
                    while a < b:
                        mid = (a + b) // 2
                        if buckets[j][1][mid].time < inst_i.time:
                            a = mid + 1
                        else:
                            b = mid

                    if i < j:
                        cc += a
                    elif i == j:
                        tr += a
                    else:
                        dc += a

        # Return C-index
        if cc + tr + dc:
            return (cc + 0.5 * tr) / (cc + tr + dc)
        else:
            return 1
    else:
        # CTree does not generate a theta, but a kaplan meier distribution in the leaf node, so when comparing to CTree, the actual distributions need to be compared.
        raise NotImplementedError()

# keep the same implementation
def calculate_cox_concordance(root, orig_instances, num_instances):
    # Group the instances together based on their theta
    buckets = {}

    for i in range(len(orig_instances)):
        orig_instance = orig_instances[i]
        num_instance = num_instances[i]
        pred = -root.get_expected_value(orig_instance, num_instance)
        if pred not in buckets:
            buckets[pred] = [[], [], []]  # [censored, death, all]
        buckets[pred][orig_instance.event].append(orig_instance)
        buckets[pred][2].append(orig_instance)
    buckets = [buckets[key] for key in sorted(buckets.keys())]
    for i in range(len(buckets)):
        for j in range(3):
            buckets[i][j] = sorted(buckets[i][j], key=lambda x: x.time)

    # Apply binary search to compute the amount of discordance and concordant pairs efficiently
    cc = tr = dc = 0
    for i in range(len(buckets)):
        for j in range(len(buckets)):
            for inst_i in buckets[i][2]:
                a, b = 0, len(buckets[j][1])
                while a < b:
                    mid = (a + b) // 2
                    if buckets[j][1][mid].time < inst_i.time:
                        a = mid + 1
                    else:
                        b = mid

                if i < j:
                    cc += a
                elif i == j:
                    tr += a
                else:
                    dc += a

    # Return C-index
    if cc + tr + dc:
        return (cc + 0.5 * tr) / (cc + tr + dc)
    else:
        return 1

# Calculates the Integrated Brier Score for a certain tree
#
# root              The tree to evaluate the instances with
# train_instances   The train instances used to create the tree
# test_instances    The train instances used to test the tree
def calculate_integrated_brier_score(root, train_instances, test_instances, method='leblanc'):
    # Drop test instances that appear after the latest train instance (ideally just a few)
    max_train_time = max(inst.time for inst in train_instances)
    test_instances = [inst for inst in test_instances if inst.time < max_train_time]

    # Get the non-extreme test-instance-times
    times = sorted(set([inst.time for inst in test_instances]))
        
    if len(times) >= 10:
        lower = times[len(times) // 10]
        upper = times[9 * len(times) // 10]
        if upper == max(times):
            upper -= 1e-3
    else:
        lower = times[0] + 1e-3
        upper = times[-1] - 1e-3
    
    if upper - lower <= 100:
        times = [lower + j * (upper - lower) / 100 for j in range(101)]
    else:
        times = np.arange(lower, upper)
    

    # Create survival distribution estimates for each test instance
    estimates = []
    estimates_per_leaf = {}
    for inst in test_instances:
        _, km_survival_distribution, leblanc_survival_distribution = root.classify(inst)
        distribution = leblanc_survival_distribution if method == 'leblanc' else km_survival_distribution

        if distribution not in estimates_per_leaf:
            estimate = []
            for t in times:
                estimate.append(distribution(t))
            estimates_per_leaf[distribution] = estimate
        estimates.append(estimates_per_leaf[distribution])

    # Format all instances to structured array
    train_instances_formatted = np.array([(inst.event, inst.time) for inst in train_instances], dtype=[("event", "?"), ("time", "f4")])
    test_instances_formatted = np.array([(inst.event, inst.time) for inst in test_instances], dtype=[("event", "?"), ("time", "f4")])

    # Use sksurv's IBS method
    score = integrated_brier_score(train_instances_formatted, test_instances_formatted, np.asarray(estimates), np.asarray(times))
    return score

def calculate_cox_integrated_brier_score(root, train_instances, original_test_instances, numeric_test_instances, method='leblanc'):
    # Drop test instances that appear after the latest train instance (ideally just a few)
    max_train_time = max(inst.time for inst in train_instances)
    test_instances = [inst for inst in numeric_test_instances if inst.time < max_train_time]
    test_instances_orig = [inst for inst in original_test_instances if inst.time < max_train_time]

    # Get the non-extreme test-instance-times
    times = sorted(set([inst.time for inst in test_instances]))

    if len(times) >= 10:
        lower = times[len(times) // 10]
        upper = times[9 * len(times) // 10]
        if upper == max(times):
            upper -= 1e-3
    else:
        lower = times[0] + 1e-3
        upper = times[-1] - 1e-3

    if upper - lower <= 100:
        times = [lower + j * (upper - lower) / 100 for j in range(101)]
    else:
        times = np.arange(lower, upper)


    # Create survival distribution estimates for each test instance
    estimates = []
    for i in range(len(test_instances)):
        original_test_instance = original_test_instances[i]
        numeric_test_instance = numeric_test_instances[i]
        pred, distr = root.cox_classify(original_test_instance, numeric_test_instance, extract_offset=True)
        def g(x):
            return pow(distr(x), exp(pred))
        distribution = g
        estimate = []
        for t in times:
            estimate.append(distribution(t))
        estimates.append(estimate)

    # Format all instances to structured array
    train_instances_formatted = np.array([(inst.event, inst.time) for inst in train_instances], dtype=[("event", "?"), ("time", "f4")])
    test_instances_formatted = np.array([(inst.event, inst.time) for inst in test_instances_orig], dtype=[("event", "?"), ("time", "f4")])

    # Use sksurv's IBS method
    score = integrated_brier_score(train_instances_formatted, test_instances_formatted, np.asarray(estimates), np.asarray(times))
    return score

def coxstreedsolve(algorithm):
    print(f"\n\033[33;1mEvaluating {algorithm.upper()}'s output...\033[0m")

    # Survival distribution method
    sd_method = "kaplan_meier"  # 'kaplan_meier' if algorithm == 'ctree' else 'leblanc'

    # Read trees
    f = open(f"{DIRECTORY}/output/{algorithm}_trees.csv")
    lines = f.read().strip().split("\n")
    f.close()

    new_lines = [";".join(lines[0].split(";")[:-2] + ["results"])]
    for line in lines[1:]:
        # Parse line
        null = None
        id, settings, time_duration, flat_tree = [eval(j) for j in line.split(";")]
        train_filename = settings["file"]
        test_filename = settings["test-file"]

        results = {}

        # Parse given tree and determine labels using training set
        tree = parse_cox_tree(flat_tree)
        train_instances, _ = fill_cox_tree(tree, f"{ORIGINAL_DIRECTORY}/{train_filename}.txt", f"{NUMERIC_DIRECTORY}/{train_filename}.txt")
        original_test_instances = read_dataset(f"{ORIGINAL_DIRECTORY}/{test_filename}.txt")
        numeric_test_instances = read_dataset(f"{NUMERIC_DIRECTORY}/{test_filename}.txt")
        base_tree = Tree(None, None, None, train_instances)

        # Store time
        results["runtime"] = time_duration

        # Count number of nodes
        results["num_nodes"] = tree.size()

        # Calculate the Integrated Brier Score ratio
        base_tree_ibs = calculate_integrated_brier_score(base_tree, train_instances, original_test_instances,
                                                         method='kaplan_meier')
        curr_tree_ibs = calculate_cox_integrated_brier_score(tree, train_instances, original_test_instances, numeric_test_instances)
        ibs_ratio = 1
        if base_tree_ibs > 1e-6:
            ibs_ratio = 1 - curr_tree_ibs / base_tree_ibs
        if abs(ibs_ratio) < 1e-6:
            ibs_ratio = 0
        results["integrated_brier_score_ratio"] = ibs_ratio

        # For both the training set and the testing set
        for name, filename, instances in [("train", train_filename, train_instances),
                                          ("test", test_filename, original_test_instances)]:
            orig_instances, num_instances = fill_cox_tree(tree, f"{ORIGINAL_DIRECTORY}/{filename}.txt", f"{NUMERIC_DIRECTORY}/{filename}.txt")

            # Fill tree and base tree with instances
            tree.clear_instances()
            predictions = []
            for i in range(len(instances)):
                original_instance = orig_instances[i]
                numerical_instance = num_instances[i]
                pred, _ = tree.cox_classify(original_instance, numerical_instance, True)
                predictions.append(pred)
            times = [o.time for o in orig_instances]
            events = [o.event for o in orig_instances]
            tree.calculate_cox_error(times, events, predictions)

            base_tree.clear_instances()
            for inst in instances:
                base_tree.classify(inst, True)
            base_tree.calculate_error()

            print(tree)

            # Calculate the objective score
            base_tree_error = base_tree.error
            tree_error = tree.error
            objective_score = 1
            if base_tree_error > 1e-6:
                objective_score = 1 - tree_error / base_tree_error
            if abs(objective_score) < 1e-6:
                objective_score = 0

            # Calculate Harrell's C-index
            concordance_score = calculate_cox_concordance(tree, orig_instances, num_instances)

            results[name] = {
                "objective_score": objective_score,
                "concordance_score": concordance_score,
            }

        # Push results on a single line
        info_line = ";".join(line.split(";")[:-2])
        results_line = str(results)
        new_line = f"{info_line};{results_line}"
        new_lines.append(new_line)
        print(f"\033[35;1m{info_line}\033[30;1m;\033[34;1m{results_line}\033[0m")

    # Write results to file
    f = open(f"{DIRECTORY}/output/{algorithm}_output.csv", "w")
    f.write("\n".join(new_lines))
    f.close()

def main():
    coxstreedsolve('coxstreedloglike')
    coxstreedsolve('coxstreedcindex')
    for algorithm in ["streed", "ctree", "ost"]:
        print(f"\n\033[33;1mEvaluating {algorithm.upper()}'s output...\033[0m")

        # Survival distribution method
        sd_method = "leblanc"#'kaplan_meier' if algorithm == 'ctree' else 'leblanc'

        # Read trees
        f = open(f"{DIRECTORY}/output/{algorithm}_trees.csv")
        lines = f.read().strip().split("\n")
        f.close()

        new_lines = [";".join(lines[0].split(";")[:-2] + ["results"])]

        for line in lines[1:]:
            # Parse line
            null = None
            id, settings, time_duration, flat_tree = [eval(j) for j in line.split(";")]
            train_filename = settings["file"]
            test_filename = settings["test-file"]

            results = {}

            # Parse given tree and determine labels using training set
            tree = parse_tree(flat_tree)
            train_instances = fill_tree(tree, f"{ORIGINAL_DIRECTORY}/{train_filename}.txt")
            test_instances = read_dataset(f"{ORIGINAL_DIRECTORY}/{test_filename}.txt")
            base_tree = Tree(None, None, None, train_instances)

            # Store time
            results["runtime"] = time_duration

            # Count number of nodes
            results["num_nodes"] = tree.size()

            # Calculate the Integrated Brier Score ratio
            curr_tree_ibs = calculate_integrated_brier_score(tree, train_instances, test_instances, method=sd_method)
            base_tree_ibs = calculate_integrated_brier_score(base_tree, train_instances, test_instances, method='kaplan_meier')
            ibs_ratio = 1
            if base_tree_ibs > 1e-6:
                ibs_ratio = 1 - curr_tree_ibs / base_tree_ibs
            if abs(ibs_ratio) < 1e-6:
                ibs_ratio = 0
            results["integrated_brier_score_ratio"] = ibs_ratio

            # For both the training set and the testing set
            for name, filename, instances in [("train", train_filename, train_instances), ("test", test_filename, test_instances)]:
                instances = fill_tree(tree, f"{ORIGINAL_DIRECTORY}/{filename}.txt")

                # Fill tree and base tree with instances
                for t in [tree, base_tree]:
                    t.clear_instances()
                    for inst in instances:
                        t.classify(inst, True)
                    t.calculate_error()

                print(tree)

                # Calculate the objective score
                base_tree_error = base_tree.error
                tree_error = tree.error
                objective_score = 1
                if base_tree_error > 1e-6:
                    objective_score = 1 - tree_error / base_tree_error
                if abs(objective_score) < 1e-6:
                    objective_score = 0

                # Calculate Harrell's C-index
                concordance_score = calculate_concordance(tree, instances, method=sd_method)

                results[name] = {
                    "objective_score": objective_score,
                    "concordance_score": concordance_score,
                }

            # Push results on a single line
            info_line = ";".join(line.split(";")[:-2])
            results_line = str(results)
            new_line = f"{info_line};{results_line}"
            new_lines.append(new_line)
            print(f"\033[35;1m{info_line}\033[30;1m;\033[34;1m{results_line}\033[0m")

        # Write results to file
        f = open(f"{DIRECTORY}/output/{algorithm}_output.csv", "w")
        f.write("\n".join(new_lines))
        f.close()

    print("\033[32;1mDone!\033[0m")

if __name__ == "__main__":
    main()
