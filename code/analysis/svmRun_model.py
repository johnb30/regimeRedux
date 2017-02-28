import os
from joblib import Parallel, delayed
import multiprocessing

from sklearn.externals import joblib

import numpy as np
from scipy.sparse import csr_matrix, hstack

from sklearn.metrics import precision_score as getPrec
from sklearn.metrics import recall_score as getRecall
from sklearn.metrics import f1_score as getF1
from sklearn.metrics import accuracy_score as getAcc
from sklearn.metrics import classification_report as classScore

from bldTrTe import buildData, flatten


def _get_data(dir_path, path):
    """Private function to get the absolute path to the installed files."""
    cwd = os.path.abspath(os.path.dirname(__file__))
    joined = os.path.join(dir_path, path)
    out_dir = os.path.join(cwd, joined)
    return out_dir


def prStats(modelName, grams, actual, pred):
    print modelName + ' grams' + str(grams[0]) + '_' + str(grams[1])
    print '\t\tPrecision: ' + str(getPrec(actual, pred))
    print '\t\tRecall: ' + str(getRecall(actual, pred))
    print '\t\tF1: ' + str(getF1(actual, pred))
    print '\t\tAccuracy: ' + str(getAcc(actual, pred))
    print '\t\t' + modelName + ' Class Level:'
    print classScore(actual, pred)


def infFeatures(path, filename, vectorizer, model, n=20):
    fNames = vectorizer.get_feature_names()
    coefs = model.coef_.transpose()
    coefData = np.random.random_integers(0, 100, n * 2).reshape(n * 2, 1)
    for lab in range(0, coefs.shape[1]):
        coefsFtr = sorted(zip(coefs[:, lab], fNames))
        posCoefsFtr = [[x[0], x[1], 'pos'] for x in coefsFtr[:-(n + 1):-1]]
        negCoefsFtr = [[x[0], x[1], 'neg'] for x in coefsFtr[:n]]
        coefsFtr = np.vstack((posCoefsFtr, negCoefsFtr))
        coefData = np.append(coefData, coefsFtr, axis=1)
    coefData = coefData[:, 1:]
    cols = ['coef' + str(x) + ',ftr' + str(x) + ',sign' + str(x)
            for x in range(1, coefs.shape[1] + 1)]
    cols = ','.join(cols) + '\n'
    filename = os.path.join(path, filename)
    with open(filename, 'wb') as f:
        f.write(b'' + cols)
        np.savetxt(f, coefData, delimiter=',', fmt="%s")


def runAnalysis(testFilename, labelFilename,
                labelCol, labelName,
                trainYr=1999, testYr=2009, grams=(2, 5),
                addWrdCnt=False, addCntry=False):

    # Load data
    testData = buildData(
        textFile=testFilename, sYr=testYr,
        labelFile=labelFilename)
    ####

    # Divide into train and test and convert
    # to appropriate format

    loadpath = _get_data('../../results', 'models')

    print('Loading tfidf')
    vec_file = '{}_tfidf.pkl'.format(labelName)
    vec_file = os.path.join(loadpath, vec_file)
    vectorizer = joblib.load(vec_file)

    xTest = vectorizer.transform(testData[:, 1])
#    yTest = np.array([int(x) for x in list(testData[:, labelCol])])

#    # Add other features
#    if(addWrdCnt):
#        wTest = csr_matrix(np.array(list(testData[:, 2]))).transpose()
#
#        xTest = hstack((xTest, wTest))
#
#    if(addCntry):
#        from pandas import factorize
#
#        cntryYr = [x.split('_')[0] for x in testData[:, 0]]
#        cntryYr = factorize(cntryYr)[0]
#        cTest = csr_matrix(np.array(list(cntryYr))).transpose()
#
#        xTest = hstack((xTest, cTest))
#    #####

    # Run SVM with linear kernel
    print('Loading SVM')
    svm_class_file = '{}_svm_class.pkl'.format(labelName)
    svm_class_file = os.path.join(loadpath, svm_class_file)
    svmClass = joblib.load(svm_class_file)
    yConfSVM = list(svmClass.decision_function(xTest))
    yPredSVM = svmClass.predict(xTest)

    print('Loading SVM2')
    svm_class2_file = '{}_svm_class2.pkl'.format(labelName)
    svm_class2_file = os.path.join(loadpath, svm_class2_file)
    svmClass_2 = joblib.load(svm_class2_file)
    yProbSVM = svmClass_2.predict_proba(xTest)

    #####
    # Print data with prediction
    testCntry = np.array([[x.split('_')[0].replace(',', '')]
                          for x in list(testData[:, 0])])
    testYr = np.array([[x.split('_')[1]] for x in list(testData[:, 0])])

    vDat = np.array([[x] for x in flatten([
        ['test'] * testData.shape[0]])])

#    testLab = np.array([[x] for x in flatten([
#        [-99] * testData.shape[0]])])
    testLab = np.array([[x] for x in list(testData[:, labelCol])])

    if labelName[0:6] == 'polCat':
        probSVM = [';'.join(['%s' % x for x in row]) for row in yProbSVM]
        confSVM = [';'.join(['%s' % x for x in sublist])
                   for sublist in yConfSVM]
    if labelName[0:6] != 'polCat':
        probSVM = yProbSVM[:, 1]
        confSVM = yConfSVM

    predSVM = np.array(yPredSVM, ndmin=2).T
    probSVM = np.array(probSVM, ndmin=2).T
    confSVM = np.array(confSVM, ndmin=2).T

    output = np.hstack((
        testCntry,
        testYr,
        vDat,
        testLab,
        np.hstack((confSVM, probSVM, predSVM))
    ))

    outpath = _get_data('../../results', 'predictions')
    outCSV = os.path.join(outpath, '{}_predictions.csv'.format(labelName))
#    outCSV = outName.replace('.txt', '.csv')
#    outCSV = os.path.join(outpath, outCSV)
    with open(outCSV, 'wb') as f:
        f.write(
            b'country,year,data,' +
            labelName +
            ',confSVM,probSVM,predSVM\n')
        np.savetxt(f, output, delimiter=',', fmt="%s")

    # Print top features for classes from SVM
    #infFeatures(outpath, outName.replace('.txt', '._wrdFtr.csv'),
    #            vectorizer, svmClass, 500)
#####

# Set up function inputs for parallized run
demTestFile = 'all_data_wdow0.json'
demLabelFile = 'demData_99-13.csv'
demTestYear = 1999
demLabelCol = 7
demLabelName = 'polGe7'
demGram = (2, 4)

mmpTestFile = 'all_data_wdow0.json'
mmpLabelFile = 'mmppData_99-10.csv'
mmpTestYear = 1999
mmpLabelCol1 = 3
mmpLabelName1 = 'monarchy'
monGram = (1, 3)
mmpLabelCol2 = 4
mmpLabelName2 = 'military'
milGram = (1, 1)
mmpLabelCol3 = 5
mmpLabelName3 = 'party'
parGram = (1, 1)
mmpLabelCol4 = 6
mmpLabelName4 = 'personal'
perGram = (2, 4)

params = [
#    (demTestFile,
#     demLabelFile,
#     demTestYear,
#     demLabelCol,
#     demLabelName,
#     demGram),
    (mmpTestFile,
     mmpLabelFile,
     mmpTestYear,
     mmpLabelCol1,
     mmpLabelName1,
     monGram),
    (mmpTestFile,
     mmpLabelFile,
     mmpTestYear,
     mmpLabelCol2,
     mmpLabelName2,
     milGram),
    (mmpTestFile,
     mmpLabelFile,
     mmpTestYear,
     mmpLabelCol3,
     mmpLabelName3,
     parGram),
    (mmpTestFile,
     mmpLabelFile,
     mmpTestYear,
     mmpLabelCol4,
     mmpLabelName4,
     perGram)
]


# Run analysis
#numCores = multiprocessing.cpu_count()
#results = Parallel(n_jobs=numCores, verbose=100)(
#    delayed(runAnalysis)(
#        testFilename=x[0], labelFilename=x[1],
#        testYr=x[2], labelCol=x[3], labelName=x[4], grams=x[5])
#    for x in params
#)

for x in params:
    print('Running {}'.format(x[4]))
    runAnalysis(testFilename=x[0], labelFilename=x[1], testYr=x[2],
                labelCol=x[3], labelName=x[4], grams=x[5])
