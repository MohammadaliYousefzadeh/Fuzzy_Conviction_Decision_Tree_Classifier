import pandas as pd
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import _tree
# -----------------------------------------------------------------------------------------------------

# Defines a sigmoid function, a common function in machine learning for mapping values to a range between 0 and 1.
# It's modified here with additional parameters: theta, fuzzy_nearly_one, and anchor_point.
# These parameters adjust the function to compute fuzzy membership values.
def sigmoid(x, theta, fuzzy_nearly_one, anchor_point):
    tau_zero = 1/np.log(fuzzy_nearly_one/(1 - fuzzy_nearly_one)) 
    tau = (anchor_point-theta) * tau_zero
    return 1 / (1 + np.exp(-(x-theta)/tau))
# -----------------------------------------------------------------------------------------------------    

# This function finds the parent of each node in a decision tree. It traverses the tree and stores the parent-child relationships.
# This information is useful for understanding the path that data takes through the tree.
def find_parent(tree):
    tree_ = tree.tree_ 
    parents = [() for _ in range(tree_.node_count)]

    def traverse(node, parents, bloodline):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            parents[node] = tuple(bloodline + [(1, node)])
            
            parents = traverse(tree_.children_left[node], parents, bloodline+[(1, node)])
            parents = traverse(tree_.children_right[node], parents, bloodline+[(0, node)])
            
            return parents
        else:
            parents[node] = tuple(bloodline + [(-1, -1)])
            
            return parents
        
    parents = traverse(0, parents, [])
        
    return parents
# -----------------------------------------------------------------------------------------------------

# Calculates fuzzy membership functions for each node in the decision tree.
# The function iterates through the tree nodes, applying the sigmoid function to compute the fuzzy membership for the passed and failed conditions at each node based on the dataset x.

def cal_fuzzy_membership_fn(tree, x, fuzzy_nearly_one):
    # x = x.reset_index(drop=True)
    tree_ = tree.tree_
    passed_fuzzy_membership_fns = [lambda x: np.nan for _ in range(tree_.node_count)]
    failed_fuzzy_membership_fns = [lambda x: np.nan for _ in range(tree_.node_count)]

    def recurse(node, passed_fns, failed_fns, x):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            threshold = tree_.threshold[node]
            
            passed_x = x.loc[x.iloc[:, tree_.feature[node]] <= threshold]
            failed_x = x.loc[x.iloc[:, tree_.feature[node]] > threshold]
            if passed_x.shape[0] > 1:
                mean = passed_x.iloc[:, tree_.feature[node]].mean()
                params = {'theta': threshold, 'fuzzy_nearly_one': fuzzy_nearly_one, 
                          'anchor_point': mean}
                passed_fns[node] = partial(sigmoid, **params)
                
                mean = failed_x.iloc[:, tree_.feature[node]].mean()
                params = {'theta': threshold, 'fuzzy_nearly_one': fuzzy_nearly_one, 
                          'anchor_point': mean}
                failed_fns[node] = partial(sigmoid, **params)
            else:
                passed_fns[node] = lambda x:(x <= threshold).astype(float)
                failed_fns[node] = lambda x:(x > threshold).astype(float)
            
            passed_fns, failed_fns = recurse(tree_.children_left[node], passed_fns, failed_fns, passed_x)
            passed_fns, failed_fns = recurse(tree_.children_right[node], passed_fns, failed_fns, failed_x)
        else:
            passed_fns[node] = lambda x:x/x
            passed_fns[node] = lambda x:x/x
            
        return passed_fns, failed_fns

    return recurse(0, passed_fuzzy_membership_fns, failed_fuzzy_membership_fns, x)
# -----------------------------------------------------------------------------------------------------

# Calculates the conviction rate for each data point in the test dataset.
# It uses the find_parent and cal_fuzzy_membership_fn functions to compute fuzzy memberships and then applies these to the test data.
# It adds several new columns to a dataframe, including fuzzy values and probabilities for each node.
def cal_conviction_rate(model, x_train, x_test, y_test, features, fuzzy_nearly_one, pivot_threshold):
    traversed_path = find_parent(model)
    passed_fuzzy_membership_fns, failed_fuzzy_membership_fns = cal_fuzzy_membership_fn(model, x_train, fuzzy_nearly_one)

    df_temp = pd.DataFrame(x_test, columns=features)
    df_temp['Target'] = y_test.values

    for i in range(model.tree_.node_count):
        df_temp['fv_{}_{}'.format(i, 0)] = failed_fuzzy_membership_fns[i](x_test.iloc[:, model.tree_.feature[i]])
        df_temp['fv_{}_{}'.format(i, 1)] = passed_fuzzy_membership_fns[i](x_test.iloc[:, model.tree_.feature[i]])

    for i, path in enumerate(traversed_path):
        df_temp['fm_{}'.format(i)] = 1
        for is_pass_condition, parent_node in path:
            if is_pass_condition >= 0:
                df_temp['fm_{}'.format(i)] *= df_temp['fv_{}_{}'.format(parent_node, is_pass_condition)]

    temp = model.tree_.decision_path(x_test.values.astype(np.float32))
    decision_path = temp.toarray()

    leaves = [i for i, path in enumerate(traversed_path) if (-1, -1) in path]
    arr_leaves = np.zeros(model.tree_.node_count)
    arr_leaves[leaves] = 1

    fuzzy_memberships = df_temp[['fm_{}'.format(i) for i in range(model.tree_.node_count)]].values

    df_temp['leaf_number'] = np.argmax(decision_path * arr_leaves, axis=1)
    df_temp['leaf_fuzzy_value'] = np.max(fuzzy_memberships * decision_path * arr_leaves, axis=1)
    df_temp['max_fuzzy_number'] = np.argmax(fuzzy_memberships * arr_leaves, axis=1)
    df_temp['max_fuzzy_value'] = np.max(fuzzy_memberships * arr_leaves, axis=1)
    df_temp['prob'] = model.predict_proba(x_test)[:, 1]

    df_lower = df_temp[df_temp['prob'] < pivot_threshold]
    df_lower = df_lower.sort_values(['prob', 'leaf_fuzzy_value'], ascending=[False, True])

    df_upper = df_temp[df_temp['prob'] >= pivot_threshold]
    df_upper = df_upper.sort_values(['prob', 'leaf_fuzzy_value'], ascending=False)

    df_temp = pd.concat([df_upper, df_lower]).reset_index(drop=True)

    return df_temp
# -----------------------------------------------------------------------------------------------------

# This function visualizes the decision tree. It uses Graphviz to create a graphical representation of the tree,
# adjusting the width of the arrows based on the number of samples that pass through each node.
def render_tree(dot_data, name=None, model_path=None):
    min_width = 1
    max_width = 20

    lines  = dot_data.split('\n')
    weight = {}
    new_lines = []
    for line in lines:
        items = line.split('[')
        if len(items) > 1 and '[label=' in line:
            node         = int(line.split(' [label="')[0])
            samples      = int(line.split('samples = ')[-1].split('\\n')[0])
            weight[node] = samples

            if node == 0:
                n = samples

            first_exp = line.split('label="')[-1].split('\\n')[0]
            if first_exp.split(' ')[0] != 'gini':
                items = first_exp.split(' ')
                k, v = items[0], items[-1]

                line = '{}label="{}\\n{}\\n{}'.format(line.split('label="')[0], node, first_exp, 
                                                        '\\n'.join(line.split('\\n')[1:]))
            else:
                line = '{}label="{}\\n{}'.format(line.split('label="')[0], node, '\\n'.join(line.split('\\n')[1:]))

        elif '->' in line:
            s, d = line.split(';')[0].split('[')[0].strip().split('->')
            s = s.strip()
            d = int(d.strip())

            if 'headlabel' in line:
                items = line.split(']')
                line  = '{}, penwidth={}] ;'.format(items[0], weight[d]/n*(max_width - min_width) + min_width)
            else:
                line = '{} [penwidth={}] ;'.format(line[:-1], weight[d]/n*(max_width - min_width) + min_width)

        new_lines.append(line)

    graph = graphviz.Source('\n'.join(new_lines))
    if model_path is not None and name is not None: 
        graph.render(name, model_path, cleanup=True, format='png')
    
    return graph
# -----------------------------------------------------------------------------------------------------

# Plots the conviction rate against the model's score.
# It creates a horizontal bar plot where the length and color of each bar represent the conviction rate and the classification result, respectively.
def plot_score_conviction_rate(df_temp, path):
    plt.figure(figsize=(10,15))

    x = np.arange(df_temp.shape[0])

    tick_lab = ['1.0', '0.5', '0.0', '0.5', '1.0']
    tick_val = [-1, -.5, 0, .5, 1]

    pivot_threshold = 0.5

    df_lower = df_temp[df_temp['prob'] < pivot_threshold]
    df_lower = df_lower.sort_values(['prob', 'leaf_fuzzy_value'], ascending=[False, True]).reset_index(drop=True)

    df_upper = df_temp[df_temp['prob'] >= pivot_threshold]
    df_upper = df_upper.sort_values(['prob', 'leaf_fuzzy_value'], ascending=False)

    df_temp = pd.concat([df_upper, df_lower]).reset_index(drop=True)

    plt.figure(figsize=(10,15))

    x = np.arange(df_temp.shape[0])

    tick_lab = ['1.0', '0.5', '0.0', '0.5', '1.0']
    tick_val = [-1, -.5, 0, .5, 1]

    colors = np.where(df_temp['Target'], 'r', 'g')[::-1]
    plt.barh(x, -df_temp['prob'].values[::-1], alpha=.75, height=1, align='center' , color=colors)
    plt.barh(x, df_temp['leaf_fuzzy_value'].values[::-1], alpha=.75, height=1, align='center', color=colors)
    plt.plot(np.zeros(x.shape[0]), x, color='w')
    # plt.yticks(x, x)
    plt.xticks(tick_val, tick_lab)
    plt.grid(b=False)
    plt.title("Population Pyramid")
    plt.savefig(path)
# -----------------------------------------------------------------------------------------------------

# Summary
# The code is designed to augment a decision tree classifier with fuzzy logic.
# It calculates fuzzy membership degrees and conviction rates, which can be used to enhance the tree's decision-making process.
# This approach could potentially improve the classifier's ability to distinguish between classes, especially in cases with uncertainty or overlap between classes.
# Visualization functions (render_tree and plot_score_conviction_rate) are provided for interpreting the model's structure and the results of the fuzzy enhancement.
# This is an advanced application and requires a good understanding of both decision trees and fuzzy logic.
# The code is highly specific and seems tailored for a particular dataset or type of problem.
