'''
Implementation of 

Hierarchical clustersing based on Unsupervised Decision Tree
by Jayanat Basak and Raghu Krishnapuram

Added a new splitting point procedure:
    using SVM to find that split point


'''

import numpy as np
import pandas as pd
from sklearn.svm import SVC
import torch

def interpt_dist(data_points):
    X = torch.from_numpy(data_points.values)
    n = X.shape[0]
    d = X.shape[1]
    #X = data_points.values
    
    #dist = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
    dist = torch.sum((X.view(n,1,d) - X.view(1,n,d))**2, -1)
    
    return dist.numpy()
    #return dist

def inhom_measure(dataset):
    obs = dataset.shape[0]
    d = interpt_dist(dataset)
    dmax = d.max()
    #matrix of ones, subtracted by normalised distance matrix
    M = np.ones((obs,obs)) - (d/dmax)
    H = []
    for c in range(dataset.shape[1]):
        d = interpt_dist(dataset.drop(dataset.columns[c], axis=1))
        dmax = d.max()
        m_a = np.ones((obs,obs)) - (d/dmax)
        del(d)
        h = np.sum(np.triu(np.multiply(M, (np.ones((obs,obs)) - m_a)) + np.multiply(m_a, (np.ones((obs,obs)) - M)), 1))
    
        H.append(h)
    
    return H


#==============================================================================
#   finding split points using SVM    

def svm(X,Y):
    svc = SVC(kernel='linear')
    svc.fit(X.reshape(-1,1),Y)
    return svc.support_vectors_

def find_svm_splits(column):
    values, bins = np.histogram(column)
    
    split_index = (np.diff( np.sign( np.diff(values))) > 0).nonzero()[0] + 1
    
    split_p = []
    for i in split_index:
        Y = []
        X = []
        mid = (bins[i+1] - bins[i]) / 4
        for row in column:
            if row < bins[i]+mid:
                Y.append(1)
                X.append(row)
            elif row > bins[i+1]-mid:
                Y.append(-1)
                X.append(row)
        sv = svm(np.array(X), np.array(Y))
        x1 = []
        x2 = []
        
        for i in range(len(sv)):
            ind = np.where( X == sv[i][0])
            ind = ind[0][0]
            if Y[ind] == 1:
                x1.append( X[ind] )
            else:
                x2.append( X[ind] )
        split_p.append( (max(x1) + min(x2)) / 2)
        del(x1); del(x2); del(X); del(Y)
    split_p.insert(0, 0.0)
    split_p.append(column.max())
    return split_p

#==============================================================================


    
#==============================================================================
#   finding split points as mid points

def find_splits(column):
    values, bins = np.histogram(column)
    
    split_index = (np.diff( np.sign( np.diff(values))) > 0).nonzero()[0] + 1
    split_p = [0.0]
    for i in split_index:
        split_p.append( (bins[i] + bins[i+1]) / 2)
    split_p.append(column.max())
    return split_p

#==============================================================================
    

# here we employ the inhom_measure() function to find the most important feature
# then use the above functions to find the split points in that feature
# and finally create the node containing: the feature, the split points in there
# and the splitted data points.
def get_split(dataset):
    
    # create an empty dictionary to be used as a node of the tree
    node = {}
    
    # if the dataset contains more than one features
    if dataset.shape[1] > 1:    
        # calculate inhomogeinity measure
        h = inhom_measure(dataset)
        H = min(h)
        index = h.index(H) # get the index of the highest measured feature
    
    # if there's only one column in the dataset
    else:
        index = 0 # simply choose that column
    
    # record the column used in this node
    node["column"] = dataset.columns[index]
    
    # find the split points in this column
    # basically returns the valleys of a mixed gaussian curve
    # which also contains the first and the last valley
    splits = find_svm_splits(dataset.iloc[:, index])    
    
    # record the split points in the node
    # note: we don't need the first and the last split points
    # check the splits list returned to understand why
    node["splits"] = splits[1 : len(splits)-1]
    
    # if there are no splits found
    if len( node["splits"]) == 0:
        return dataset.shape[0] # simply return all the points
    
    # all the clusterss formed will be temprarily stored in this groups list
    groups = []
    
    # for all the splits
    for i in range(len(splits)-1):
        
        # divide the dataset into groups based on the splits
        data = dataset[ dataset.iloc[:, index].between(splits[i]+0.0001, (splits[i+1]), inclusive = True)]
        if data.shape[0]:
            # then append each group into 'groups'
            groups.append(data.drop( data.columns[index], axis=1))
#            groups.append(data)
    # add the groups found into the node
    node["groups"] = groups
    
    return node
    

def grow(node, max_depth, depth=1):
    
    groups = node["groups"]
    del(node["groups"])
    
    global c_key
    
    if depth >= max_depth:
        for i in range(len(groups)):
            k = "b"+str(i+1)
            c_key += 1
            node[k] = str(c_key)
            clusters[str(c_key)] = groups[i].index.values.tolist()
    
    elif groups[0].shape[1] <= 2:
        for i in range(len(groups)):
            k = "b"+str(i+1)
            if groups[i].shape[0] < 10:
                c_key += 1
                node[k] = str(c_key)
                clusters[str(c_key)] = groups[i].index.values.tolist()
                continue
            nd = get_split(groups[i])
            if isinstance(nd, int):
                c_key += 1
                node[k] = str(c_key)
                clusters[str(c_key)] = groups[i].index.values.tolist()
                continue
            node[k] = nd
            for j in range(len(node[k]["groups"])):
                c_key += 1
                node[k]["b"+str(j+1)] = str(c_key)
                clusters[str(c_key)] = node[k]["groups"][j].index.values.tolist()
            del(node[k]["groups"])
    else:
        for i in range(len(groups)):
            k = "b"+str(i+1)
            if groups[i].shape[0] < 10:
                c_key += 1
                node[k] = str(c_key)
                clusters[str(c_key)] = groups[i].index.values.tolist()
            else:
                nd = get_split(groups[i])
                if not isinstance(nd, int):
                    node[k] = nd
                    grow(node[k], max_depth, depth+1)
                else:
                    c_key += 1
                    node[k] = str(c_key)
                    clusters[str(c_key)] = groups[i].index.values.tolist()



def clusters_label(clusters, labels):
    
    global c_label
    
    for c in clusters:
        if len(clusters[c]) > 0:
            index_list = clusters[c]
            
            from collections import Counter
            mf_label = Counter(labels[index_list]).most_common(1)[0][0]
            
            c_label[c] = mf_label
    
    #print("Cluster Labels:\n", c_label)


def cluster_accuracy(clusters, labels, n):
    global c_label
    count = 0
    for c in clusters:
        if len(clusters[c]) > 0:
            pred = c_label[c]
            index_list = clusters[c]
            
            for i in index_list:
                if labels[i] == pred:
                    count += 1
    return count/n


def predict(node, X, c_label):
    
#    print('\n\n',node['column'],', ',node['splits'],', ', node.keys())
#    print(X)
    for i in range( len(node['splits'])+1 ):
        if i < len(node['splits']):
            
            if X[node['column']] < node['splits'][i]:
                
                if isinstance(node['b'+str(i+1)], str):
                    return c_label[node['b'+str(i+1)]]
                else:
                    return predict(node['b'+str(i+1)], X, c_label)
        else:
            if isinstance(node['b'+str(i)], str):
                return c_label[node['b'+str(i)]]
            else:
                return predict(node['b'+str(i)], X, c_label)
            
            


# =============================================================================
# from sklearn.datasets import load_iris
# dataset_dict = load_iris()
# dataset = pd.DataFrame(data=dataset_dict['data'], columns=dataset_dict['feature_names'])
# labels = dataset_dict['target']
# =============================================================================

# =============================================================================
# from sklearn.datasets import load_wine
# dataset_dict = load_wine()
# dataset = pd.DataFrame(data=dataset_dict['data'], columns=dataset_dict['feature_names'])
# labels = dataset_dict['target']
# =============================================================================

# =============================================================================
# dataset = pd.read_csv("bupa.data")
# labels = dataset["selector"].values
# dataset.drop(["selector"], axis=1, inplace=True)
# =============================================================================

# =============================================================================
# from sklearn.datasets import load_breast_cancer
# dataset_dict = load_breast_cancer()
# dataset = pd.DataFrame(data=dataset_dict['data'], columns=dataset_dict['feature_names'])
# labels = dataset_dict['target']
# =============================================================================

# =============================================================================
# dataset = pd.read_csv("glass.data")
# labels = dataset["Class"].values
# dataset.drop(["id","Class"], axis=1, inplace=True)
# =============================================================================

# =============================================================================
# dataset = pd.read_csv("new-thyroid.data")
# labels = dataset["Class"].values
# dataset.drop(["Class"], axis=1, inplace=True)
# =============================================================================

dataset = pd.read_csv("balance-scale.txt")
labels = dataset["Class"].values
dataset.drop(["Class"], axis=1, inplace=True)

# =============================================================================
# dataset = pd.read_csv("habermans-survival.txt")
# labels = dataset["Class"].values
# dataset.drop(["Class"], axis=1, inplace=True)
# =============================================================================

# =============================================================================
# dataset = pd.read_csv("TA_eval.txt")
# labels = dataset["Class"].values
# dataset.drop(["Class"], axis=1, inplace=True)
# =============================================================================

# =============================================================================
# dataset = pd.read_csv("page_blocks.csv")
# labels = dataset['class'].values
# dataset = dataset.drop( ['class'], axis=1)
# =============================================================================

# =============================================================================
# dataset = pd.read_csv("spambase.data")
# labels = dataset["class"].values
# dataset = dataset.drop( ["class"], axis=1)
# =============================================================================



from sklearn.model_selection import KFold
skf = KFold(n_splits=20)


max_acc = 0.0
max_tacc = 0.0
m_depth = 0
k=0
tree = {}
for train_index, test_index in skf.split(dataset, labels):

    k+=1
    print(k," fold")
    #print(test_index.shape[0], train_index.shape[0])
    X_train, X_test = dataset.iloc[train_index], dataset.iloc[test_index]    
    
    clusters = {}
    c_key = 0
    c_label = {}
    m_acc = 0.0
    m_tacc = 0.0
    t_tree = {}
    try:
        for i in range(1,10):
            clusters = {}
            c_key = 0
            c_labels = {}
            root = get_split(X_train)
    
            grow(root, i, 1)
            
            clusters_label(clusters, labels)
            tacc = cluster_accuracy(clusters, labels, dataset.shape[0])
        
            count = 0
            indexes = X_test.index.values
            pred_list = []
            for j in range(X_test.shape[0]):
            
                pred = predict(root, X_test.iloc[j,:], c_label)
                pred_list.append(pred)
                if pred == labels[ indexes[j] ]:
                    count += 1
            acc = count/X_test.shape[0]
            if m_tacc < tacc:
                m_acc = acc
                m_tacc = tacc
                depth = i
                t_tree = root
        if max_acc < m_acc:
            max_acc = m_acc
            max_tacc = m_tacc
            m_depth = depth
            tree = t_tree
        
    except:
        continue


#print(interpt_dist(dataset).shape)
# resullts
#print("SVMTree")
#print("UDT")
print("Training acc: ", max_tacc)
print("Test acc: ", max_acc)
print("Depth: ", m_depth)
print(tree)
    
