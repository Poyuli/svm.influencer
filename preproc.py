import math
import numpy
    
def lin2log(prob_x):
    """
    Transform original data instances x into log scales.
    """
    
    new_x = []
    for item in prob_x:
        xi = {}
        for i in range(1, len(item) + 1):
            if item[i] == 0: 
                val = 0.1
            else:
                val = item[i]
            xi[i] = math.log( val )
        new_x += [xi]
    return new_x

def ratio_f(prob_x, remove):
    """
    Add additional features to data instances x.
    Ratios of follower counts to following counts will be added.
    
    If remove = True, only ratios will be kept and following/follower counts will be removed.
       remove = False, additional features as well as the original ones will be kept.
    """
    new_x = []
    for item in prob_x:
        if item[2] == 0: item[2] = 0.1
        if item[13] == 0: item[13] = 0.1
        xi = {}
        if remove == True:
            item[1] = item[1] / item[2]
            item[12] = item[12] / item[13]
            elem_list = item.values()
            elem_list.pop(1)
            elem_list.pop(11)
            for i in range(1,len(elem_list) + 1):
                xi[i] = elem_list[i-1]
        else:
            a = item[1] / item[2]
            b = item[12] / item[13]
            elem_list = item.values()
            elem_list.insert(11, a)
            elem_list.insert(23, b)
            for i in range(1,len(elem_list) + 1):
                xi[i] = elem_list[i-1]
            
        new_x += [xi]
    return new_x
    
def ratio_ab(prob_x):
    """
    Reduce all features of data instances x to ratios of A's features to B's.
    Make sure that all features of prob_x are placed in following order:
    
    prob_x = {1: a1, 2: a2,..., n: an, n+1: b1, n+2: b2,..., 2n: bn}
    where ai and bi are i-th feature of a and b, respectively.
    
    """
    
    new_x = []
    for item in prob_x:
        xi = {}
        d = len(item)/2
        for i in range(1, d+1):
            if item[i + d] == 0:
                val = item[i] / 0.1
            else:
                val = item[i] / item[i + d]
            xi[i] = val
        new_x += [xi]
    return new_x
    
def svm_scale(prob_x, scale):
    """
    svm_scale(input_feature_vector, scaling method) -> x
    
    Read LIBSVM-format input vector to scale data with different scaling methods.
    Put scale = 'signed' to scale data within (-1, 1)
                'unsigned' to scale data within (0, 1)
                'normal' to scale data centered around the mean and normalized by standard deviation
                
    This function is a practice without using libraries such as scikit-learn.
    """
    
    max_feat = {}
    min_feat = {}
    mean_feat = {}
    sd_feat = {}
    new_x = []
    
    for i in range(1, len(prob_x[0])+1):
        max_feat[i] = max(item[i] for item in prob_x)
        min_feat[i] = min(item[i] for item in prob_x)
        mean_feat[i] = sum(item[i] for item in prob_x) / len(prob_x)
        sd_feat[i] = math.sqrt( sum( (item[i] - mean_feat[i]) **2 for item in prob_x) / (len(prob_x) - 1) )  
        
    for i in range(0, len(prob_x)):
        scale_inner = {}
        for j in range(1,len(prob_x[0])+1):
            if scale == "unsigned":
                scale_inner[j] = ( prob_x[i][j] - min_feat[j] ) / ( max_feat[j] - min_feat[j] )
            if scale == "signed":
                scale_inner[j] = ( 2*prob_x[i][j] - min_feat[j] - max_feat[j] ) / ( max_feat[j] - min_feat[j] )
            if scale == "normal":
                scale_inner[j] = ( prob_x[i][j] - mean_feat[j] ) / sd_feat[j]
        new_x += [scale_inner]           
    return new_x
    
def normalize(prob_x):
    """
    Normalize all data instances to unit norms.
    """
    
    new_x = []
    for item in prob_x:
        xi = {}
        elem = item.values()
        elem /= numpy.linalg.norm(elem)
        for i in range(0, len(elem)):
            xi[i+1] = elem[i]
        new_x += [xi]
    return new_x
    
def fselect(prob_x, flist):
    """
    Select features with the list of feature indices.
    
    If flist = [1,3,4], 1st, 3rd, 4th features will be selected.
    """
    
    new_x = []
    for item in prob_x:
        xi = {}
        for i in item.keys():
            if i in flist:
                xi[i] = item[i]
            else:
                continue
        new_x += [xi]
    return new_x





        


