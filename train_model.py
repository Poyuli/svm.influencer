import os
import sys
import subset
import math
from fileproc import *
from preproc import *
from grid import *
from svmutil import *

# Initialization
file_path = "/Users/BradLi/Documents/libsvm-3.18/python"
file_name = "train.csv"
feat_list = [1,3,4,5,9,12,14,15,16,20]
scale_factor = "unsigned"


# Read the data from csv file
os.chdir(file_path)
y, x, header = svm_read_csv(file_name)


# Feature engineering
# Remove the feature engineering of ratio as it does not impact the accuracy.
#x = ratio_f(x, remove = True)
#x = ratio_ab(x)


# svm_scale, normalize, lin2log work with the same accuracy as they perform similar action.
x = lin2log(x)
x = svm_scale(x, scale_factor)
x = normalize(x)
x = fselect(x, feat_list)
write2txt(x, y, "scaled_data.txt")


# Data sampling for training
#sys.argv = ["subset.py", "scaled_data.txt", "3500", "cv_data.txt", "test_data.txt"]
#subset.main(sys.argv)


# Read labels y and instances x
ytrain, xtrain = svm_read_problem("scaled_data.txt")
#ytest, xtest = svm_read_problem("test_data.txt")


# Find optimal g (RBF kernel parameter) and c (regularization parameter) with coarse and fine grid search
#rate, param = find_parameters("scaled_data.txt","-log2g -5,15,2 -log2c 3,-15,-2")
param = {}
param["c"] = 8
param["g"] = 8
c = math.log(param["c"], 2) - 1
g = math.log(param["g"], 2) - 1
rate, param = find_parameters("scaled_data.txt","-log2g " + str(c) + "," + str(c+2) + ",0.25 -log2c " + str(g+2) + "," + str(g) + ",-0.25")


# Train the SVM model with optimized g and c
model = svm_train(ytrain, xtrain, "-g " + str(param["g"]) + " -c " + str(param["c"]))
train_labs, train_acc, train_vals = svm_predict(ytrain, xtrain, model)
#test_labs, test_acc, test_vals = svm_predict(ytest, xtest, model)



        


