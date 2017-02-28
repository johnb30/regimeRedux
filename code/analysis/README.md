Analysis
========

Scripts used to analyze the scraped and cleaned data.

* `bldTrTe.py` constructs the necessary array out of the `X` (text) and `y` (label) data. Requires modification depending on whether in-sample or true out-of-sample work is being done.
* `svmRun_model.py` assumes the presence of pre-trained TFIDF and SVM models. This script is used to save time on running and to do true out-of-sample work, i.e., the situation where there is no labeled training data.
* `svmRun.py` is the workhorse script. This trains the TFIDF and SVM models.
* `nn/` contains files for the experimental work with `doc2vec` and neural net models, including a simple CNN and a CNN -> LSTM architecture.
