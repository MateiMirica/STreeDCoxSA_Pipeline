from math import log, exp, sqrt
import os
import warnings
import numpy as np

DIRECTORY = os.path.realpath(os.path.dirname(__file__))
ORIGINAL_DIRECTORY = f"{DIRECTORY}/datasets/original"
NUMERIC_DIRECTORY = f"{DIRECTORY}/datasets/numeric"
BINARY_DIRECTORY = f"{DIRECTORY}/datasets/binary"
INDICES_DIRECTORY = f"{DIRECTORY}/datasets/indices"

# Reads the settings from a filename and returns them as maps
# The file must be formatted with one JSON object on each individual line
#
# filename      The path to the settings file
def parse_settings(filename):
    f = open(filename)
    settings = [eval(j) for j in f.read().strip().split("\n")]
    f.close()

    return settings

def parse_value(value):
    for op in [int, float, str]:
        try:
            return op(value)
        except:
            pass
    return None

def parse_line(line, sep=","):
    return [parse_value(j) for j in line.split(sep)]

def files_in_directory(directory):
    return [j for j in os.listdir(directory) if os.path.isfile(f"{directory}/{j}")]

def nelson_aalen(instances):
    ts = {}
    for inst in instances:
        t = inst.time
        d = inst.event

        if t not in ts:
            ts[t] = [0, 0]
        ts[t][d] += 1

    #d = {0: 0}
    d = {0: 1 / (len(instances) + 1)}
    at_risk = len(instances)
    sum = 0
    for t in sorted(ts.keys()):
        left, died = ts[t]
        sum += died / at_risk
        at_risk -= died + left

        if died > 0:
            d[t] = sum

    keys = [*d.keys()]

    def f(x):
        try:
            return d[x]
        except:
            pass

        a = 0
        b = len(keys) - 1
        while a != b:
            mid = (a + b + 1) // 2
            if keys[mid] > x:
                b = mid - 1
            else:
                a = mid
        return d[keys[a]]

    return f

def kaplan_meier(instances):
    ts = {}
    for inst in instances:
        t = inst.time
        d = inst.event

        if t not in ts:
            ts[t] = [0, 0]
        ts[t][d] += 1

    d = {0: 1}
    at_risk = len(instances)
    prev_t = 0
    for t in sorted(ts.keys()):
        left, died = ts[t]
        d[t] = d[prev_t] * (1 - died / at_risk)
        at_risk -= died + left
        prev_t = t

    keys = [*d.keys()]
    def f(x):
        try:
            return d[x]
        except:
            pass

        a = 0
        b = len(keys) - 1
        while a != b:
            mid = (a + b + 1) // 2
            if keys[mid] > x:
                b = mid - 1
            else:
                a = mid
        return d[keys[a]]

    return f

def leblanc(hazard_function, theta):
    def f(t):
        return exp(-theta * hazard_function(t))
    return f

class Instance:
    def __init__(self, feats):
        self.time = feats.pop("time")
        self.event = int(float(feats.pop("event")) > 0.5)
        self.feats = feats

    def __repr__(self):
        return f"<t = {self.time}, d = {self.event}, feats = {self.feats}>"

def calculate_theta(events, hazards):
    if len(events) == 0:
        warnings.warn("encountered empty leaf node.")
        return 1

    numerator = max(0.5, sum(events))
    denominator = sum(hazards)

    theta = numerator / denominator
    return theta

class Tree:
    def __init__(self, criterium, tree_0, tree_1, instances=None):
        self.criterium = criterium
        self.trees = [tree_0, tree_1] if tree_0 else []
        self.instances = instances or []

        self.x_scale = []
        self.x_offset = []
        self.coefs = []
        self.baseline_survival = []
        self.unique_times = []
        self.model_offset = None
        self.theta = None
        self.kaplan_meier_distribution = None
        self.leblanc_distribution = None
        self.breslow_distribution = None
        self.error = None

        if instances:
            self.calculate_label()
            self.calculate_error()

    def size(self):
        if not self.trees:
            return 0
        return self.trees[0].size() + 1 + self.trees[1].size()

    def classify(self, instance, store=False):
        if store:
            self.instances.append(instance)

        if not self.trees:
            return (self.theta, self.kaplan_meier_distribution, self.leblanc_distribution)

        if not self.criterium(instance.feats):
            return self.trees[0].classify(instance, store)
        else:
            return self.trees[1].classify(instance, store)

    def calculate_leaf_breslow(self):
        predictions = []
        times = []
        events = []
        if not self.trees:
            if self.breslow_distribution != None:
                return
            x = []
            for inst in self.instances:
                x.append(list(inst.feats.values()))
            x_offset = []
            x_scale = []
            nrows = len(x)
            ncols = len(x[0])
            for i in range(ncols):
                x_offset.append(0)
                x_scale.append(0)
            for i in range(nrows):
                for j in range(ncols):
                    x_offset[j] += x[i][j]
            for j in range(ncols):
                x_offset[j] /= nrows
            for i in range(nrows):
                for j in range(ncols):
                    x[i][j] -= x_offset[j]
            for i in range(nrows):
                for j in range(ncols):
                    x_scale[j] += (x[i][j] ** 2)
            for j in range(ncols):
                x_scale[j] = sqrt(x_scale[j])
                if x_scale[j] == 0:
                    x_scale[j] = 1
            # for i in range(nrows):
            #     for j in range(ncols):
            #         x[i][j] /= x_scale[j]
            self.model_offset = 0
            for i in range(ncols):
                self.model_offset += x_offset[i] * self.coefs[i]
            self.x_scale = x_scale
            self.x_offset = x_offset
            for i in range(len(self.instances)):
                inst = self.instances[i]
                pred = 0
                for j in range(len(self.coefs)):
                    pred += self.coefs[j] * x[i][j]
                predictions.append(pred)
                times.append(inst.time)
                events.append(inst.event)
            self.breslow_distribution, x, y = breslow(predictions, events, times)
            self.unique_times = x
            self.baseline_survival = y
        else:
            self.trees[0].calculate_leaf_breslow()
            self.trees[1].calculate_leaf_breslow()

    def get_expected_value(self, original_instance, numeric_instance):
        if not self.trees:
            s = 0
            for i in range(len(self.coefs)):
                s += self.coefs[i] * list(numeric_instance.feats.values())[i]
            s -= self.model_offset
            area = 0
            for i in range(len(self.unique_times)):
                if i == 0:
                    area += self.unique_times[i]
                else:
                    area += pow(self.baseline_survival[i - 1], exp(s)) * (self.unique_times[i] - self.unique_times[i - 1])
            half_area = area / 2.0
            area = 0
            for i in range(len(self.unique_times)):
                prev_area = area
                if i == 0:
                    area += self.unique_times[i]
                else:
                    area += pow(self.baseline_survival[i - 1], exp(s)) * (self.unique_times[i] - self.unique_times[i - 1])
                if area >= half_area:
                    rem_area = half_area - prev_area
                    if i == 0:
                        return rem_area
                    return rem_area / pow(self.baseline_survival[i - 1], exp(s)) + self.unique_times[i - 1]

        if not self.criterium(original_instance.feats):
            return self.trees[0].get_expected_value(original_instance, numeric_instance)
        else:
            return self.trees[1].get_expected_value(original_instance, numeric_instance)

    def cox_classify(self, original_instance, numeric_instance, store=False, extract_offset=False):
        if store:
            self.instances.append(numeric_instance)

        if not self.trees:
            s = 0
            for i in range(len(self.coefs)):
                s += self.coefs[i] * list(numeric_instance.feats.values())[i]
            if extract_offset and self.model_offset is not None:
                s -= self.model_offset
            return s, self.breslow_distribution

        if not self.criterium(original_instance.feats):
            return self.trees[0].cox_classify(original_instance, numeric_instance, store, extract_offset)
        else:
            return self.trees[1].cox_classify(original_instance, numeric_instance, store, extract_offset)

    def calculate_label(self):
        events = [inst.event for inst in self.instances]
        hazards = [Tree.hazard_function(inst.time) for inst in self.instances]

        self.theta = calculate_theta(events, hazards)
        self.kaplan_meier_distribution = kaplan_meier(self.instances)

        return (self.theta, self.kaplan_meier_distribution)

    def calculate_leblanc_km_estimator(self, hazard_function):
        if self.trees:
            for i in range(2):
                self.trees[i].calculate_leblanc_km_estimator(hazard_function)
            return
        
        assert(self.theta is not None)
        self.leblanc_distribution = leblanc(hazard_function, self.theta)

    def calculate_error(self):
        if self.trees:
            self.error = 0
            for i in range(2):
                if self.trees[i].error == None:
                    self.trees[i].error = self.trees[i].calculate_error()
                self.error += self.trees[i].error
            return self.error

        if self.theta == None or self.kaplan_meier_distribution == None:
            self.theta, self.kaplan_meier_distribution = self.calculate_label()

        event_sum = 0
        negative_log_hazard_sum = 0

        for inst in self.instances:
            event = inst.event
            hazard = Tree.hazard_function(inst.time)

            if event:
                event_sum += 1
                negative_log_hazard_sum += -log(hazard)
        
        self.error = max(0, negative_log_hazard_sum - event_sum * log(self.theta))

        return self.error


    def calculate_cox_error(self, times, events, predictions):
        if self.trees:
            self.error = 0
            for i in range(2):
                if self.trees[i].error == None:
                    self.trees[i].error = self.trees[i].calculate_cox_error(times, events, predictions)
                self.error += self.trees[i].error
            return self.error

        self.error = 0
        # times = np.array(times)
        # events = np.array(events)
        # predictions = np.array(predictions)
        # order = np.argsort(-times, kind="mergesort")
        # times = times[order]
        # events = events[order]
        # predictions = predictions[order]
        # part_sum = 0
        # log_like = 0
        # i = 0
        # while i < len(predictions):
        #     t = times[i]
        #     j = i
        #     while i < len(predictions) and times[i] == t:
        #         part_sum += exp(predictions[i])
        #         i+=1
        #     while j < i:
        #         if events[j]:
        #             log_like += (predictions[j] - log(part_sum))
        #         j+=1
        #
        # self.error = -log_like / len(predictions)

        return self.error

    def get_leaves(self):
        if self.trees:
            return self.trees[0].get_leaves() + self.trees[1].get_leaves()
        return [self]

    def clear_instances(self):
        self.error = None
        self.instances = []
        for child in self.trees:
            child.clear_instances()

    def to_string(self, path):
        trace = "".join(["\033[30m:\033[0m     ", "║     ", "╚═\033[31;1mX\033[0m═══", "╠═\033[32;1mV\033[0m═══"][path[j] + 2 * (j == len(path) - 1)] for j in range(len(path)))

        r = []
        if not self.trees:
            theta_repr = "???"
            if self.theta != None:
                theta_repr = f"{self.theta:.4f}"
            elif self.coefs != []:
                theta_repr = f"{self.coefs}"
            r.append(f"{trace}\033[34;1m(\033[36m{theta_repr}\033[34m)\033[0m ")
        else:
            r.append(f"{trace}\033[43;38;2;0;0;0m X \033[0m ")
            path.append(1)
            r.append(self.trees[1].to_string(path))
            path[-1] = 0
            r.append(self.trees[0].to_string(path))
            path.pop()

        if self.instances:
            if self.trees:
                r[0] += "\033[38;5;232m"
            else:
                r[0] += "\033[38;5;248m"
            r[0] += f"<error = {self.error or 0:.4f}, avg-error = {(self.error or 0) / len(self.instances):.4f}, {sum(inst.event for inst in self.instances)} / {len(self.instances)} died>\033[0m"
        r[0] += "\n"
        return "".join(r)

    def __repr__(self):
        return self.to_string([]).strip()

def get_feature_meanings(filename):
    f = open(f"{DIRECTORY}/datasets/feature_meanings/{filename}.txt")
    lines = f.read().strip().split("\n")
    f.close()

    feature_meanings = {}
    for line in lines:
        idx = line.find(" = ")
        key = line[:idx]
        value = line[idx + 3:]
        feature_meanings[key] = value

    return feature_meanings

def read_dataset(filename):
    f = open(filename)
    lines = f.read().strip().split("\n")
    f.close()

    keys, lines = lines[0].split(","), lines[1:]

    instances = []
    for values in [parse_line(line, sep=",") for line in lines]:
        inst = Instance({key: value for key, value in zip(keys, values)})
        instances.append(inst)

    return instances

def parse_tree(d):
    if len(d) == 1:
        tree = Tree(-1, None, None)
        tree.theta = d[0]
        return tree
    else:
        feat = d[0]
        tree_0 = parse_tree(d[1])
        tree_1 = parse_tree(d[2])
        return Tree(feat, tree_0, tree_1)

def is_array(lst):
    return all(isinstance(x, (int, float)) for x in lst)

def parse_cox_tree(d):
    if is_array(d) == 1:
        tree = Tree(-1, None, None)
        tree.coefs = d
        return tree
    else:
        feat = d[0]
        tree_0 = parse_cox_tree(d[1])
        tree_1 = parse_cox_tree(d[2])
        return Tree(feat, tree_0, tree_1)

def read_tree(filename):
    f = open(filename)
    dataset_filename, d = f.read().strip().split("\n")
    f.close()

    null = None
    flat_tree = eval(d)
    return parse_tree(flat_tree), dataset_filename

def fill_tree(tree, dataset_filename):
    instances = read_dataset(dataset_filename)

    Tree.hazard_function = nelson_aalen(instances)

    for inst in instances:
        tree.classify(inst, True)
    
    tree.calculate_error()
    tree.calculate_leblanc_km_estimator(Tree.hazard_function)

    return instances

def _compute_counts(event, time, order=None):
    n_samples = event.shape[0]

    if order is None:
        order = np.argsort(time, kind="mergesort")

    uniq_times = np.empty(n_samples, dtype=time.dtype)
    uniq_events = np.empty(n_samples, dtype=int)
    uniq_counts = np.empty(n_samples, dtype=int)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    times = np.resize(uniq_times, j)
    n_events = np.resize(uniq_events, j)
    total_count = np.resize(uniq_counts, j)
    n_censored = total_count - n_events

    # offset cumulative sum by one
    total_count = np.r_[0, total_count]
    n_at_risk = n_samples - np.cumsum(total_count)

    return times, n_events, n_at_risk[:-1], n_censored

def breslow(prediction, event, time):

    prediction = np.array(prediction)
    event = np.array(event)
    time = np.array(time)
    order = np.argsort(time, kind="mergesort")
    risk_score = np.exp(prediction)
    risk_score = risk_score[order]
    uniq_times, n_events, n_at_risk, _ = _compute_counts(event, time, order)

    divisor = np.empty(n_at_risk.shape, dtype=float)
    value = np.sum(risk_score)
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[k : (k + d)].sum()
        k += d
        divisor[i] = value
        if divisor[i] == 0:
            ok = 1

    assert k == n_at_risk[0] - n_at_risk[-1]

    y = np.cumsum(n_events / divisor)
    y = np.exp(-y)

    def f(x):
        last = 0
        a = 0
        b = len(y) - 1
        while a <= b:
            mid = (a + b) // 2
            if uniq_times[mid] <= x:
                last = mid
                a = mid + 1
            else:
                b = mid - 1
        return y[last]

    return f, uniq_times, y

def fill_cox_tree(tree, original_dataset_filename, numeric_dataset_filename):
    o_instances = read_dataset(original_dataset_filename)
    n_instances = read_dataset(numeric_dataset_filename)
    s_times = np.array([o.time for o in o_instances])
    order = np.argsort(-s_times, kind="mergesort")

    original_instances = []
    for i in range(len(o_instances)):
        original_instances.append(o_instances[order[i]])

    numeric_instances = []
    for i in range(len(n_instances)):
        numeric_instances.append(n_instances[order[i]])

    predictions = []
    for i in range(len(original_instances)):
        pred, _ = tree.cox_classify(original_instances[i], numeric_instances[i], True)
        predictions.append(pred)

    times = [o.time for o in original_instances]
    events = [o.event for o in original_instances]

    tree.calculate_leaf_breslow()
    tree.calculate_cox_error(times, events, predictions)
    Tree.hazard_function = nelson_aalen(original_instances)

    return original_instances, numeric_instances
