import numpy as np

# author: Loginaway #

class Node(object):
    '''
    Basic element for decision tree.

    Initialization: to initialize a tree node, at least feature_name is needed.

    Connection: to connect different nodes, there are two situations.
        1, if the parent node is discrete (i.e. node.discrete==True), 
            then 
            >> node[feature_value]=childnode 
            will do the connection.
        2, if the parent node is continuous, then 
            >> node['<=']=childnode 
            or 
            >> node['>']=childnode 
            will assign its children. To be specific, 
            node['<=']=childnode means if the feature value is bigger than 
            node.threshold, then the workflow goes left to the childnode, and vice versa.
    
    Classification: if you have constructed a tree, you may want to classify a 
        certain group of data. Simply call the root node will do this task, i.e.
        >> root(data)
        where data should be a dictionary containing the corresponding key and value.
        And this method will return the classification result.

    Visualization: A naive visualization is implemented here. 
        For example, you may visualize a subtree rooted at 'node' by
        >> print(node)
        The tree structure will then be printed on the console.

    '''
    def __init__(self, feature_name='', discrete=True, threshold=0, isLeaf=False, classification=None):
        # for discrete node: the next node can be retrieved by self.map[feature_value]
        # for continuous node: by self.map['<='] (when value <= self.threshold)
        #                       by self.map['>'] (when value > self.threshold)
        self.map=dict()

        self.discrete=discrete
        self.feature_name=feature_name
        self.isLeaf=isLeaf
        if isLeaf:
            self.classification=classification
        if not discrete:
            self.threshold=threshold

    def __setitem__(self, key, value):
        self.map[key]=value

    def __getitem__(self, key):
        return self.map.get(key)

    def __call__(self, data):
        '''
        This method should be used on the root to predict its classification.
        data: a dict with its key being the features and value being 
        the corresponding value.
        '''
        if self.isLeaf:
            return self.classification
        # if the node has a discrete feature, use __getitem__ to find the next node
        # then call the next node.
        if self.discrete:
            return self.map[data[self.feature_name]](data)
        else:
            if data[self.feature_name].astype(np.float32)>self.threshold:
                return self.map['>'](data)
            else:
                return self.map['<='](data)

    def __str__(self):
        '''
        This method is designed for printing the subtree rooted at the current node.
        To print the tree, try print(self).
        '''
        hierarchy={}
        stack=[(0, 0, self)]
        count=0
        while stack:
            # BFS
            (layer_index, node_index, current)=stack.pop(0)
            layer_index+=1

            hierarchy[layer_index]=hierarchy.get(layer_index, [])
            hierarchy[layer_index].append((node_index, current.feature_name,\
                    [(str(current.classification),)] if current.isLeaf else \
                     [(str(key), '' if current.discrete else str(current.threshold), item.feature_name) for key, item in current.map.items()]))
            # print(hierarchy)

            if current.isLeaf:
                continue
            for _, item in current.map.items():
                count+=1
                stack.append((layer_index, count, item))
            # print(item_index_map)

        totallist=[]
        for layer, layer_content in hierarchy.items():
            rowlist=[]
            for i in layer_content:
                rowlist.append('['+i[1]+']'+': {'+', '.join([' '.join(item) for item in i[2]])+'}')
            rowstr=' +++ '.join(rowlist)
            totallist.append(rowstr)

        return '============Root===========\n'+'\n\n'.join(totallist)+'\n============Leaf==========='
