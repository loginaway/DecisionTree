from DecisionTree.Node import Node
import numpy as np
import sys
# set the maximal recursion limits here.
sys.setrecursionlimit(10000)

# author: Loginaway #

class baseClassDecisionTree(object):
    '''
    The main class of decision tree.
    '''

    def __init__(self, feature_discrete=[], treeType='C4.5'):
        '''
        feature_discrete: a dict with its each key-value pair being (feature_name: True/False),
            where True means the feature is discrete and False means the feature is 
            continuous. 
        type: ID3/C4.5/CART
        pruning: pre/post
        '''
        self.feature_discrete=feature_discrete
        self.treeType=treeType
        self.leaf_count=0
        self.tmp_classification=''

        self.tree=None

    def Entropy(self, list_of_class):
        '''
        Compute the entropy for the given list of class.
        list_of_class: an array of classification labels, e.g. ['duck', 'duck', 'dolphin']
        'duck': 2/3, 'dolphin': 1/3, so the entropy for this array is 0.918
        '''
        count={}
        for key in list_of_class:
            count[key]=count.get(key, 0)+1
        frequency=np.array(tuple(count.values()))/len(list_of_class)
        return -1*np.vdot(frequency, np.log2(frequency))

    def Information_Gain(self, list_of_class, grouped_list_of_class):
        '''
        Compute the Information Gain.
        list_of_class: an array of classification labels, e.g. ['duck', 'duck', 'dolphin']
        grouped_list_of_class: the list of class grouped by the values of 
            a certain attribute, e.g. [('duck'), ('duck', 'dolphin')].
        The Information_Gain for this example is 0.2516.
        '''
        sec2=np.sum([len(item)*self.Entropy(item) for item in grouped_list_of_class])/len(list_of_class)
        return self.Entropy(list_of_class)-sec2

    def Information_Ratio(self, list_of_class, grouped_list_of_class):
        '''
        Compute the Information Ratio.
        list_of_class: an array of classification labels, e.g. ['duck', 'duck', 'dolphin']
        grouped_list_of_class: the list of class grouped by the values of 
            a certain attribute, e.g. [('duck'), ('duck', 'dolphin')].
        The Information_Ratio for this example is 0.2740.
        '''
        tmp=np.array([len(item)/len(list_of_class) for item in grouped_list_of_class])
        intrinsic_value=-1*np.vdot(tmp, np.log2(tmp))
        return self.Information_Gain(list_of_class, grouped_list_of_class)/intrinsic_value

    def Gini(self, list_of_class):
        '''
        Compute the Gini value.
        list_of_class: an array of classification labels, e.g. ['duck', 'duck', 'dolphin']
        The Gini value for this example is 0.4444.
        '''
        count={}
        for key in list_of_class:
            count[key]=count.get(key, 0)+1
        prob=np.array(tuple(count.values()))/len(list_of_class)
        return 1-np.vdot(prob, prob)

    def Gini_Index(self, list_of_class, grouped_list_of_class):
        '''
        Compute the Gini Index.
        list_of_class: an array of classification labels, e.g. ['duck', 'duck', 'dolphin']
        grouped_list_of_class: the list of class grouped by the values of 
            a certain attribute, e.g. [('duck'), ('duck', 'dolphin')].
        The Gini Index for this example is 0.3333.
        '''
        return np.sum([len(item)*self.Gini(item) \
            for item in grouped_list_of_class])/len(list_of_class)

    def orderByGainOrRatio(self, D, A, by='Gain'):
        '''
        Return the order by Information Gain or Information Ratio.
        by: 'Gain', 'Ratio'.
        For the definition of D and A, see the remark in method 'fit'.
        '''
        tmp_value_dict=dict()
        target_function=self.Information_Gain if by=='Gain' else self.Information_Ratio

        for attr, info in A.items():
            possibleVal=np.unique(D[:, info[0]])
            # if the continuous attribute have only one possible value, then 
            # choosing it won't improve the model, so we abandon it.
            if len(possibleVal)==1:
                continue

            if self.feature_discrete[attr] is True:
                # discrete
                if len(info)<2:
                    A[attr].append(possibleVal)
                # retrieve the grouped list of class
                grouped_list_of_class=[]
                for val in possibleVal:
                    indexes=np.argwhere(D[:, info[0]]==val)
                    grouped_list_of_class.append(D[indexes, -1].flatten())
                IC_value=target_function(D[:, -1], grouped_list_of_class)
                tmp_value_dict[attr]=IC_value
            else:
                # continuous
                cut_points=(possibleVal[: -1].astype(np.float32)+possibleVal[1:].astype(np.float32))/2
                maxMetric=-1
                for point in cut_points:
                    smaller_set=D[np.argwhere(D[:, info[0]]<=str(point)), -1].flatten()
                    bigger_set=D[np.argwhere(D[:, info[0]]>str(point)), -1].flatten()
                    # compute the metric
                    IC_tmp=target_function(D[:, -1], (smaller_set, bigger_set))
                    if IC_tmp>maxMetric:
                        maxMetric=IC_tmp
                        threshold=point
                # set the threshold
                if len(info)<2:
                    A[attr].append(threshold)
                else:
                    A[attr][1]=threshold
                tmp_value_dict[attr]=maxMetric
        # find the attribute with the max tmp_value_dict value
        attr_list=list(tmp_value_dict.keys())
        attr_list.sort(key=lambda x: tmp_value_dict[x])
        return attr_list

    def orderByGiniIndex(self, D, A):
        '''
        Return the order by Gini Index.
        For the definition of D and A, see the remark in method 'fit'.
        '''
        tmp_value_dict=dict()

        for attr, info in A.items():
            possibleVal=np.unique(D[:, info[0]])
            # if the continuous attribute have only one possible value, then 
            # choosing it won't improve the model, so we abandon it.
            if len(possibleVal)==1:
                continue

            if self.feature_discrete[attr] is True:
                # discrete
                if len(info)<2:
                    A[attr].append(possibleVal)
                # retrieve the grouped list of class
                grouped_list_of_class=[]
                for val in possibleVal:
                    indexes=np.argwhere(D[:, info[0]]==val)
                    grouped_list_of_class.append(D[indexes, -1].flatten())
                GI_value=self.Gini_Index(D[:, -1], grouped_list_of_class)
                tmp_value_dict[attr]=GI_value
            else:
                # continuous
                cut_points=(possibleVal[: -1].astype(np.float32)+possibleVal[1:].astype(np.float32))/2
                minMetric=9999999999
                for point in cut_points:
                    smaller_set=D[np.argwhere(D[:, info[0]]<=str(point)), -1].flatten()
                    bigger_set=D[np.argwhere(D[:, info[0]]>str(point)), -1].flatten()
                    # compute the metric
                    GI_tmp=self.Gini_Index(D[:, -1], (smaller_set, bigger_set))
                    if GI_tmp<minMetric:
                        minMetric=GI_tmp
                        threshold=point
                # set the threshold
                if len(info)<2:
                    A[attr].append(threshold)
                else:
                    A[attr][1]=threshold
                tmp_value_dict[attr]=minMetric
        # return the attribute list sorted by tmp value
        attr_list=list(tmp_value_dict.keys())
        attr_list.sort(key=lambda x: tmp_value_dict[x])
        return attr_list

    def chooseAttribute(self, D, A):
        '''
        Choose an attribute from A according to the metrics above.
        For the definition of D and A, see method 'fit'.

        Different principal for different tree types:
        ID3: choose the attribute that maximizes the Information Gain.
        C4.5: 
            1, choose those attributes whose Information Gain are above average.
            2, choose the one that maximizes the Gain Ratio from these attributes.
        CART: choose the attribute that minimizes the Gini Index.
        '''
        if self.treeType=='ID3':
            attr_list=self.orderByGainOrRatio(D, A, by='Gain')
            return attr_list[-1]

        if self.treeType=='C4.5':
            attr_list=self.orderByGainOrRatio(D, A, by='Gain')
            # for C4.5, we choose the attributes whose Gain are above average
            # and then order them by Ratio.
            sub_A={key: A[key] for key in attr_list}
            attr_list=self.orderByGainOrRatio(D, sub_A, by='Ratio')
            return attr_list[-1]
        
        if self.treeType=='CART':
            attr_list=self.orderByGiniIndex(D, A)
            return attr_list[0]    

    def fit(self, D, A):
        '''
        Train the tree.

        To save the training result:
        >> self.tree=self.fit(D, A)

        D: the training set, a size [m, n+1] numpy array (with str type elements), 
            where m is the number of training data and n is the number of attributes.
            The last column of D is the classifications (or labels).
        A: the attributes set. It is a dict with its structure being like 
            {attr_name: [index_in_D_columns, possibleVal_or_threshold], ...}

            attr_name: name of the attribute
            index_in_D_columns: the corresponding index of the attribute in ndarray D (starting from 0)
            possibleVal_or_threshold: 
                ###################################################
                ## This value may not always be available in A   ##
                ## it is added after 'chooseAttribute' is called ##
                ## And it will be updated after each call        ##
                ###################################################
                1, if the attribute is discrete, then it is a ndarray containing all possible values 
                    of this attribute.
                2, if the attribute is continuous, then possibleVal_or_threshold is the most recent 
                    threshold.
        '''
        if len(D)==0:
            node=Node(feature_name='leaf-'+str(self.leaf_count), isLeaf=True, \
                classification=self.tmp_classification)
            self.leaf_count+=1
            return node

        if len(np.unique(D[:, -1]))<=1:
            node=Node(feature_name='leaf-'+str(self.leaf_count), isLeaf=True, \
                classification=D[0, -1])
            self.leaf_count+=1
            return node

        if len(A)==0 or len(np.unique(D[:, :-1], axis=0))<=1:
            count_dict={}
            for key in D[:, -1]:
                count_dict[key]=count_dict.get(key, 0)+1
            most_frequent=sorted(D[:, -1], key=lambda x: count_dict[x])[-1]

            node=Node(feature_name='leaf-'+str(self.leaf_count), isLeaf=True, \
                classification=most_frequent)
            self.leaf_count+=1
            return node 

        count_dict={}
        for key in D[:, -1]:
            count_dict[key]=count_dict.get(key, 0)+1
        most_frequent=sorted(D[:, -1], key=lambda x: count_dict[x])[-1]
        self.tmp_classification=most_frequent

        # choose target attribute
        target_attr=self.chooseAttribute(D, A)
        # print(target_attr)

        # generate nodes for each possible values of the target attribute if it's discrete
        # generate two nodes for the two classification if it's continuous
        # related information is stored in A[target_attr][1] now, 
        # since we have called chooseAttribute at least once.
        info=A[target_attr]
        if self.feature_discrete[target_attr]:
            node=Node(feature_name=target_attr, discrete=True, isLeaf=False)
            # generate nodes for each possible values
            for possibleVal in info[1]:
                keys=set(A.keys()).difference({target_attr})
                # connect node to its child
                tmp_D=D[np.argwhere(D[:, info[0]]==possibleVal), :]
                tmp_A={key: A[key] for key in keys}

                node[possibleVal]=self.fit(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), tmp_A)
        
        else:
            # continuous
            threshold=info[1]
            node=Node(feature_name=target_attr, discrete=False, threshold=threshold,isLeaf=False)

            tmp_D=D[np.argwhere(D[:, info[0]]<=str(threshold)), :]
            node['<=']=self.fit(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), A)

            tmp_D=D[np.argwhere(D[:, info[0]]>str(threshold)), :]
            node['>']=self.fit(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), A)
            
        return node

    def post_prune(self, training_D, testing_D, A, current=None, parent=None):
        '''
        self.tree is required.

        This method conducts the post-pruning to enhance the model performance.

        To make sure this method will work, set 
        >> current=self.tree
        when you call it.
        '''
        self.current_accuracy=self.evaluate(testing_D, A)
        count_dict={}
        if len(training_D)==0:
            return 
        # print(training_D)
        for key in training_D[:, -1]:
            count_dict[key]=count_dict.get(key, 0)+1
        most_frequent=sorted(training_D[:, -1], key=lambda x: count_dict[x])[-1]

        leaf_parent=True
        for key, node in current.map.items():
            if not node.isLeaf:
                leaf_parent=False
                # Recursion, DFS
                if node.discrete:
                    tmp_D=training_D[np.argwhere(training_D[:, A[current.feature_name][0]]==key), :]
                else:
                    if key=='<=':
                        tmp_D=training_D[np.argwhere(training_D[:, A[current.feature_name][0]]<=str(node.threshold)), :]
                    else:
                        tmp_D=training_D[np.argwhere(training_D[:, A[current.feature_name][0]]>str(node.threshold)), :]
                self.post_prune(tmp_D.reshape((tmp_D.shape[0], tmp_D.shape[2])), testing_D, A, parent=current, current=node)
        
        tmp_node=Node(feature_name='leaf-'+str(self.leaf_count), isLeaf=True, classification=most_frequent)
        
        if parent:
            # when current node is not the root
            for key, node in parent.map.items():
                if node==current:
                    parent.map[key]=tmp_node
                    saved_key=key
                    break
            # compare the evaluation, if it is enhanced then prune the tree.
            tmp_accuracy=self.evaluate(testing_D, A)
            if tmp_accuracy<self.current_accuracy:
                parent.map[saved_key]=current
            else:
                self.current_accuracy=tmp_accuracy
                self.leaf_count+=1
            return
        else:
            # when current node is the root
            saved_tree=self.tree
            self.tree=tmp_node
            tmp_accuracy=self.evaluate(testing_D, A)
            if tmp_accuracy<self.current_accuracy:
                self.tree=saved_tree
            else:
                self.current_accuracy=tmp_accuracy
                self.leaf_count+=1
            return

    def predict(self, D, A):
        '''
        Predict the classification for the data in D.

        For the definition of A, see method 'fit'. 
        There is one critical difference between D and that defined in 'fit':
            the last column may or may not be the labels. 
            This method works as long as the feature index in A matches the corresponding
            column in D.
        '''
        row, _=D.shape
        pred=np.empty((row, 1), dtype=str)
        tmp_data={key: None for key in A.keys()}
        for i in range(len(D)):
            for key, info in A.items():
                tmp_data[key]=D[i, info[0]]
            pred[i]=self.tree(tmp_data)
        return pred
    
    def evaluate(self, testing_D, A):
        '''
        Evaluate the performance of decision tree. (Accuracy)

        For definition of testing_D and A, see 'predict'.
        However, here the testing_D is required to be labelled, that is, its last column 
        should be labels of the data.
        '''
        true_label=testing_D[:, -1]
        pred_label=self.predict(testing_D, A)
        
        success_count=0
        for i in range(len(true_label)):
            if true_label[i]==pred_label[i]:
                success_count+=1

        return success_count/len(true_label)

        

class decisionTree(baseClassDecisionTree):
    def __init__(self, conf):
        super().__init__(conf['feature_discrete'], conf['treeType'])
        self.conf=conf

    # try rename 'train' as 'fit' and run again, what is wrong?
    def train(self, D):
        self.tree=super().fit(D, self.conf['A'])
        
    def prune(self, training_D, testing_D):
        super().post_prune(training_D, testing_D, self.conf['A'], current=self.tree)

    def pred(self, D):
        return super().predict(D, self.conf['A'])
    
    def eval(self, D):
        return super().evaluate(D, self.conf['A'])




if __name__=='__main__':

    # training_D=np.genfromtxt('../../dataset/watermelon/watermelon2.0_train.txt', skip_header=1, dtype=str)
    # testing_D=np.genfromtxt('../../dataset/watermelon/watermelon2.0_test.txt', skip_header=1, dtype=str)
    # # print(training_D, testing_D, sep='\n')
    # A={'色泽':[0], '根蒂':[1], '敲击':[2], '纹理':[3], '脐部':[4], '触感':[5]}
    # feature_discrete={'色泽':True, '根蒂':True, '敲击':True, '纹理':True, '脐部':True, '触感':True}
    # dectree=baseClassDecisionTree(feature_discrete=feature_discrete, treeType='ID3')
    # dectree.tree=dectree.fit(training_D, A)
    # print(dectree.tree)
    # print(dectree.evaluate(testing_D, A))
    # dectree.prune(training_D, testing_D, A, current=dectree.tree)
    # print(dectree.evaluate(testing_D, A))
    # print(dectree.tree)

    # training_D=np.genfromtxt('../../dataset/watermelon/watermelon3.0_train.txt', skip_header=1, dtype=str)
    # testing_D=np.genfromtxt('../../dataset/watermelon/watermelon3.0_test.txt', skip_header=1, dtype=str)
    # # print(D)
    # A={'色泽':[0], '根蒂':[1], '敲声':[2], '纹理':[3], '脐部':[4], '触感':[5], '密度':[6], '含糖率':[7]}
    # feature_discrete={'色泽':True, '根蒂':True, '敲声':True, '纹理': True, '脐部': True, '触感':True, '密度':False, '含糖率':False}
    # dectree=baseClassDecisionTree(feature_discrete=feature_discrete, treeType='ID3')
    # dectree.tree=dectree.fit(training_D, A)
    # print(dectree.tree)
    # print(dectree.evaluate(testing_D, A))
    # dectree.prune(training_D, testing_D, A, current=dectree.tree)
    # print(dectree.evaluate(testing_D, A))
    # print(dectree.tree)

    D=np.genfromtxt('../../dataset/iris/iris_processed.txt', dtype=str)
    np.random.shuffle(D)
    k=len(D)
    attribute_name=['sepal length', 'sepal width', 'petal length', 'petal width']
    A={name: [i] for i, name in enumerate(attribute_name)}
    feature_discrete={name: False for name in attribute_name}

    conf={'A': A, 'feature_discrete': feature_discrete, 'treeType':'CART'}
    dt=decisionTree(conf)
    dt.train(D[:len(D)//2])
    print(dt.tree)
    print(dt.eval(D[len(D)//2:]))
    dt.prune(D[:len(D)//2], D[len(D)//2:])
    print(dt.tree)
    print(dt.eval(D[len(D)//2:]))


