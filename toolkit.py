import numpy
import re

class toolkit:
    def readConf(self, filename='./decisionTree.conf'):
        with open(filename, 'r') as f:
            text=f.read()
        trainset_pat=re.compile(r'trainset_name=(.*)\n')
        testset_pat=re.compile(r'testset_name=(.*)\n')
        feature_discrete_pat=re.compile(r'feature_discrete=(.*)\n')
        treeType_pat=re.compile(r'treeType=(.*)\n')
        pruning_pat=re.compile(r'pruning=(.*)\n')
        save_name_pat=re.compile(r'save_name=(.*)\n')
        
        conf={}
        conf['trainset']=trainset_pat.findall(text)[0].strip()
        conf['testset']=testset_pat.findall(text)[0].strip() if testset_pat.findall(text) else ''
        conf['feature_discrete']=eval('{'+feature_discrete_pat.findall(text)[0].strip()+'}')
        conf['treeType']=treeType_pat.findall(text)[0].strip()
        conf['pruning']=eval(pruning_pat.findall(text)[0].strip())
        conf['save_name']=save_name_pat.findall(text)[0].strip()
        conf['A']={key: [i] for i, key in enumerate(conf['feature_discrete'].keys())}
        return conf

    def genfromtxt(self, filename):
        return numpy.genfromtxt(filename, dtype=str)

