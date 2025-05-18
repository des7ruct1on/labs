import numpy as np
from collections import Counter

def find_best_split(feature_values, target_labels):
    feat_vals = np.array(feature_values)
    tgt_labels = np.array(target_labels)
    
    sort_idx = np.argsort(feat_vals)
    sorted_feat = feat_vals[sort_idx]
    sorted_tgt = tgt_labels[sort_idx]
    
    diff_mask = sorted_feat[1:] != sorted_feat[:-1]
    split_points = (sorted_feat[1:] + sorted_feat[:-1]) / 2
    valid_splits = split_points[diff_mask]
    
    if not valid_splits.size:
        return np.array([]), np.array([]), None, None
    
    n_total = len(tgt_labels)
    total_pos = np.sum(tgt_labels)
    
    cum_pos = np.cumsum(sorted_tgt)[:-1][diff_mask]
    left_sizes = np.arange(1, n_total)[diff_mask]
    right_sizes = n_total - left_sizes
    
    left_pos_prob = cum_pos / left_sizes
    left_neg_prob = 1 - left_pos_prob
    
    right_pos_prob = (total_pos - cum_pos) / right_sizes
    right_neg_prob = 1 - right_pos_prob
    
    left_impurity = 1 - left_pos_prob**2 - left_neg_prob**2
    right_impurity = 1 - right_pos_prob**2 - right_neg_prob**2
    
    weighted_impurity = (left_sizes/n_total)*left_impurity + (right_sizes/n_total)*right_impurity

    best_idx = np.argmin(weighted_impurity)
    best_threshold = valid_splits[best_idx]
    best_gini = weighted_impurity[best_idx]
    
    return valid_splits, weighted_impurity, best_threshold, best_gini

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, node_data, node_labels, current_node, current_depth=0):
        if self._should_stop(node_labels, current_depth):
            self._make_terminal_node(current_node, node_labels)
            return

        best_split = self._find_optimal_split(node_data, node_labels)
        
        if best_split is None:
            self._make_terminal_node(current_node, node_labels)
            return

        self._create_internal_node(
            current_node,
            best_split['feature'],
            best_split['threshold'],
            best_split['feature_type']
        )

        left_mask, right_mask = self._get_split_masks(
            node_data[:, best_split['feature']],
            best_split['threshold'],
            best_split['feature_type']
        )

        self._fit_node(
            node_data[left_mask],
            node_labels[left_mask],
            current_node['left_child'],
            current_depth + 1
        )
        self._fit_node(
            node_data[right_mask],
            node_labels[right_mask],
            current_node['right_child'],
            current_depth + 1
        )

    def _should_stop(self, labels, depth):
        if np.all(labels == labels[0]):
            return True
        if self._max_depth is not None and depth >= self._max_depth:
            return True
        if (self._min_samples_split is not None and 
            len(labels) < self._min_samples_split):
            return True
        return False

    def _make_terminal_node(self, node, labels):
        node['type'] = 'terminal'
        node['class'] = Counter(labels).most_common(1)[0][0]

    def _find_optimal_split(self, data, labels):
        best_split = {
            'gini': float('inf'),
            'feature': None,
            'threshold': None,
            'feature_type': None
        }

        for feature_idx in range(data.shape[1]):
            feature_type = self._feature_types[feature_idx]
            feature_data = data[:, feature_idx]
            
            if feature_type == 'categorical':
                processed_data, categories_info = self._process_categorical_feature(
                    feature_data, labels
                )
            elif feature_type == 'real':
                processed_data = feature_data.astype(float)
                categories_info = None
            else:
                raise ValueError(f"Неизвестный тип признака: {feature_type}")

            _, _, threshold, gini = find_best_split(processed_data, labels)
            
            if gini is None:
                continue

            if not self._is_split_valid(processed_data, threshold):
                continue

            if gini < best_split['gini']:
                best_split.update({
                    'gini': gini,
                    'feature': feature_idx,
                    'threshold': self._format_threshold(
                        threshold, feature_type, categories_info
                    ),
                    'feature_type': feature_type
                })

        return best_split if best_split['feature'] is not None else None

    def _process_categorical_feature(self, feature_data, labels):
        value_counts = Counter(feature_data)
        positive_counts = Counter(feature_data[labels == 1])
        
        ratios = {
            cat: positive_counts.get(cat, 0) / cnt 
            for cat, cnt in value_counts.items()
        }
        sorted_cats = sorted(ratios.keys(), key=lambda x: ratios[x])
        cat_mapping = {cat: idx for idx, cat in enumerate(sorted_cats)}
        
        return np.array([cat_mapping.get(x, 0) for x in feature_data]), cat_mapping

    def _is_split_valid(self, feature_data, threshold):
        if self._min_samples_leaf is None:
            return True
            
        left_size = np.sum(feature_data < threshold)
        right_size = len(feature_data) - left_size
        
        return (left_size >= self._min_samples_leaf and 
                right_size >= self._min_samples_leaf)

    def _format_threshold(self, threshold, feature_type, categories_info):
        if feature_type == 'real':
            return threshold
        return [cat for cat, idx in categories_info.items() if idx < threshold]

    def _create_internal_node(self, node, feature, threshold, feature_type):
        node.update({
            'type': 'nonterminal',
            'feature_split': feature,
            'left_child': {},
            'right_child': {}
        })
        
        if feature_type == 'real':
            node['threshold'] = threshold
        else:
            node['categories_split'] = threshold

    def _get_split_masks(self, feature_data, threshold, feature_type):
        if feature_type == 'real':
            left_mask = feature_data.astype(float) < threshold
        else:
            left_mask = np.isin(feature_data, threshold)
        
        return left_mask, ~left_mask

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            if float(x[feature]) < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")
    
    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])