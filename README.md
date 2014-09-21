svm
===

Tools for LIBSVM in Python, including the I/O processing part, feature engineering part, and training part.


For I/O precessing fileproc.py, there is a CSV file reader which aligns data with the LIBSVM format. In addition, it can write data into a txt file.

For feature engineering part, it includes various feature engineering methods such as scaling, normalization, feature selection.

For training part, it includes data subsetting, optimization for RBF kernels and SVM training.

* Prerequisite: subset.py, grid.py, fselect.py, svmutil.py and svm.py in the LIBSVM package
