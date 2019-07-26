from DecisionTree.DecisionTree import decisionTree
from DecisionTree.toolkit import toolkit
import pickle

toolkit=toolkit()
conf=toolkit.readConf('DecisionTree/decisionTree.conf')
training_D=toolkit.genfromtxt(conf['trainset'])
testing_D=toolkit.genfromtxt(conf['testset']) if conf['testset'] else None

dt=decisionTree(conf)
dt.train(training_D)
print(dt.tree)
if testing_D is not None:
    print('Accuracy', dt.eval(testing_D))
if conf['pruning']:
    print('Start pruning...')
    dt.prune(training_D, testing_D)
    print('Pruning finished.')
    print('Accuracy', dt.eval(testing_D))
    print(dt.tree)
if conf['save_name']:
    with open(conf['save_name'], 'wb') as f:
        pickle.dump(dt, f)
        print('Structure saved to file:', conf['save_name'])
