# Decision Tree

This is a thorough implementation of ID3, C4.5 and CART decision trees. It can process both discrete and continuous data.

If you have any question, please file an issue or contact me by loginaway@gmail.com.

## Dependencies

Only *numpy* is needed to be installed manually. Try

`pip install numpy`

## Usage

##### Example Usage

1, modify the configuration in decisionTree.conf.

2, run `python -m DecisionTree` in the parent directory of DecisionTree.

##### Dataset

For starters, your dataset should be a .txt file with each row being an item of data. Each element in the row can be a string or integer or data of any form, separated by \t . The last element in each row should be the label of the item.

##### Configuration

You are required to modify decisionTree.conf if you want to personalize the settings. There are some explanations for the options.

> trainset_name *
>
> testset_name

​	The training dataset name and the testing dataset name. You can leave out the testing set name so that the program will only train the decision tree. However, it will also not be able to do evaluation or pruning.

> feature_discrete *

​	Here you are required to enter each name of the column, and whether the data in the column is discrete or not. The name of the column should be quoted by '' or "".

​	For example, if there's a continuous attribute on column 1 named 'Height', and a discrete attribute on column 2 named 'Age', then set feature_discrete as follows

​	`feature_discrete='Height': False, 'Age': True`

​	You are not required to type in the information of the last column, which is the label of data.

> treeType *

​	The type of the decision tree you want to train. Available options are ID3, C4.5 and CART, e.g.

​	`treeType=ID3 / treeType=C4.5 / treeType=CART`

> pruning 

​	whether to do post-pruning or not

​	`pruning=True / pruning=False`

> save_name

​	The filename you want to save the tree (as pickle.dump(save_name, f)).



## References

[1] Zhihua CHOU, *Machine Learning*

[2] Quinlan, *Bagging,  boosting, and C4.5*, 2006

[3] Lewis, *An Introduction to Classification and Regression Tree (CART) Analysis*, 2000

​	



