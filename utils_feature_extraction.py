# %%
from sklearn.tree import _tree
from collections import defaultdict

# %%
def extract_rules_features_from_dt(
    num_layers,
    num_neurons, 
    binary_decision_trees,
    custom_function_name,
    binary_feature_names,
    f1_threshold=0.7,
):
    dt_rules = defaultdict(dict)

    for layer in range(num_layers):
        for neuron in range(num_neurons):
            binary_tree_model = binary_decision_trees[layer][custom_function_name]['binary_decision_tree']['model'].estimators_[neuron]
            f1 = binary_decision_trees[layer][custom_function_name]['binary_decision_tree']['f1'][neuron].item()
            if (f1_threshold is not None) and f1 < f1_threshold:
                continue

            (rules, pred_strengths, samples_per_rule, _), _ = extract_and_rules(binary_tree_model, binary_feature_names, target_class=1, value_threshold=0.7)

            sorted_rules = sorted(
                zip(rules, pred_strengths, samples_per_rule),
                key=lambda x: (x[2], x[1]),  # sort by samples_per_rule, then pred_strength
                reverse=True
            )

            filter_min_samples = binary_tree_model.tree_.n_node_samples[0].item() / 59 * .05
            filtered_rules = [(rule_infer(rule), strength, samples) for rule, strength, samples in sorted_rules if samples >= filter_min_samples]

            rule_list = []
            filtered_features = set()
            filtered_direct_features = set()
            for rule, strength, samples in filtered_rules:
                # print(f"Rule: {rule}\n\t(Strength: {strength:.2f}, Samples: {samples})")
                direct_feat_infered = set(rule.split(" AND "))
                feature_inferred = {feat.split("(")[-1].split(")")[0].split(" ")[-1] for feat in direct_feat_infered}
                
                rule_list.append(rule)
                filtered_features.update(feature_inferred)
                filtered_direct_features.update(direct_feat_infered)
                
            dt_rules[layer][neuron] = {
                "dt_rules": rule_list,
                "dt_filtered_features": filtered_features,
                "dt_filtered_directional_features": filtered_direct_features,
            }

    return dt_rules

# %%
def extract_and_rules(tree, feature_names, target_class=1, min_samples=None, value_threshold=None):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    pred_strengths = []
    samples_per_rule = []
    features_per_rule = []
    used_features = set()
    
    def recurse(node, conditions, features_in_path):
        # if (tree_.feature[node] != _tree.TREE_UNDEFINED) and (tree_.n_node_samples[node] > min_samples):
        recurse_condition = (tree_.feature[node] != _tree.TREE_UNDEFINED)
        if min_samples is not None:
            recurse_condition = recurse_condition and (tree_.n_node_samples[node] > min_samples)
        if value_threshold is not None:
            values = tree_.value[node][0]
            recurse_condition = recurse_condition and (values[target_class].item() < value_threshold)

        if recurse_condition:  # not a leaf
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # left child (feature <= threshold)
            # recurse(tree_.children_left[node],
            #         conditions + [f"({name} <= {threshold:.4f})"],
            #         features_in_path | {name})
            recurse(tree_.children_left[node],
                    conditions + [f"(NOT {name})"],
                    features_in_path | {name})
            
            # right child (feature > threshold)
            # recurse(tree_.children_right[node],
            #         conditions + [f"({name} > {threshold:.4f})"],
            #         features_in_path | {name})
            recurse(tree_.children_right[node],
                    conditions + [f"({name})"],
                    features_in_path | {name})
        else:
            # Leaf node: check predicted class
            values = tree_.value[node][0]
            pred_class = values.argmax()
            if pred_class == target_class:
                rule = " AND ".join(conditions)
                rules.append(rule)
                pred_strengths.append(values[pred_class].item() / values.sum().item())
                samples_per_rule.append(tree_.n_node_samples[node].item())
                features_per_rule.append(features_in_path)
                used_features.update(features_in_path)
    
    recurse(0, [], set())
    return (rules, pred_strengths, samples_per_rule, features_per_rule), used_features

def extract_probe_features(matrices, k=2):
    matrices_mean = matrices.mean().item()
    matrices_std = matrices.std().item()

    filtered_feature_names = []
    directional_feature_names = []
    for row in range(8):
        for col in range(8):
            square = chr(ord('A') + row) + str(col)
            mine_weight = matrices[0, row, col].item()
            empty_weight = matrices[1, row, col].item()
            theirs_weight = matrices[2, row, col].item()
            flipped_weight = matrices[3, row, col].item()
            just_played_weight = matrices[4, row, col].item()

            occupied = 0
            if mine_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_mine")
                directional_feature_names.append(f"({square}_mine)")
                # occupied = 1

            if mine_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_mine")
                directional_feature_names.append(f"(NOT {square}_mine)")

            if theirs_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_theirs")
                directional_feature_names.append(f"({square}_theirs)")
                # occupied = 1
            
            if theirs_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_theirs")
                directional_feature_names.append(f"(NOT {square}_theirs)")

            if empty_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_empty")
                directional_feature_names.append(f"({square}_empty)")
            
            if empty_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_empty")
                directional_feature_names.append(f"(NOT {square}_empty)")
                # occupied = 1
            
            if flipped_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_flipped")
                directional_feature_names.append(f"({square}_flipped)")
                # occupied = 1
            
            if flipped_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_flipped")
                directional_feature_names.append(f"(NOT {square}_flipped)")

            if just_played_weight - matrices_mean > k*matrices_std:
                filtered_feature_names.append(f"{square}_just_played")
                directional_feature_names.append(f"({square}_just_played)")
                # occupied = 1
            
            if just_played_weight - matrices_mean < -k*matrices_std:
                filtered_feature_names.append(f"{square}_just_played")
                directional_feature_names.append(f"(NOT {square}_just_played)")

    return filtered_feature_names, directional_feature_names

# %%
def infer_positive_from_negations(features, group, values):
    """
    If all but one value in a group are negated, infer the positive for the last one.
    """
    feats = set(features)
    negs = {v for v in values if f"(NOT {group}_{v})" in feats}
    remaining = set(values) - negs

    if len(remaining) == 1 and len(negs) == len(values) - 1:
        # Replace negations with the positive
        inferred = f"({group}_{remaining.pop()})"
        feats = (feats - {f"(NOT {group}_{v})" for v in negs}) | {inferred}

    return feats

def remove_negations_if_positive_present(features, group, values):
    """
    If a positive feature is present, remove all negations of the same group.
    """
    feats = set(features)
    positives = [f"({group}_{v})" for v in values if f"({group}_{v})" in feats]
    
    if positives:
        # Remove all negations if any positive is present
        feats = feats - {f"(NOT {group}_{v})" for v in values}
    
    return feats

def extract_squares(directional_features):
    squares = set()
    for feat in directional_features:
        squares.add(feat.split("(")[-1].split(" ")[-1].split("_")[0])
    return squares

def direct_feature_infer(directional_features):
    squares = extract_squares(directional_features)
    for square in squares:
        directional_features = infer_positive_from_negations(directional_features, square, ["mine", "theirs", "empty"])
        directional_features = remove_negations_if_positive_present(directional_features, square, ["mine", "theirs", "empty"])
    # filtered = set()
    # for feat in features:
    #     if feat.startswith("NOT "):
    #         continue
    #     if feat.endswith("_flipped") or feat.endswith("_just_played"):
    #         continue
    #     filtered.add(feat)
    return directional_features

def rule_infer(rule):
    directional_features = set(rule.split(" AND "))
    direct_feat_inferred = direct_feature_infer(directional_features)
    rule_inferred = " AND ".join(sorted(direct_feat_inferred))
    return rule_inferred
