svm
===

Tools for LIBSVM in Python.
There are I/O processing part, feature engineering part, and training part.

For I/O precessing fileproc.py, there is a CSV file reader which alignes data with the LIBSVM format. In addition, it writes data into a txt file.

For feature engineering part, it includes various feature engineering methods such as scaling, normalization, feature selection.

For training part, it includes data subsetting, optimization for RBF kernels and SVM training.
