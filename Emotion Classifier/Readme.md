# EmotionClassifier 
**FOR ENGL 581 (Natural Language Processing)**
Fall 2022     |     Kenyon, Vrushabh, Ketaki, Shrinivas

This a supervised multi label classifier system using LinearSVM implemented in python using object-oriented methodologies. Its goal is to train based on a given labeled set and be able to predict emotion of a given sentence and print results of said prediction. 

--- 

## Installation

### Installing Libraries

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages in the import section below (if not already included, this system is functional on RC OnDemand with no additional added libraries. For proof of this see here: https://gyazo.com/9bbb6d655729a36dad2a3648a8058bb5 and here: https://gyazo.com/7dacdc4f3325f7342185feee04f6c458). Below is an example of how to install pandas:

```bash

pip install pandas
# for python3
pip3 install pandas
```
To install a libray that helps with searching for other libraries, install pip_search and use it (for example, to search for numpy packages) https://pypi.org/project/pip-search/
```bash
pip install pip_search
# example with numpy
pip_search numpy
```

### Imports used in EmotionClassifier 
``` python
import pandas as pd
import numpy as np
from scipy.stats import randint
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import itertools 
from sklearn.model_selection import train_test_split
import re
import tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, wordnet
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
import tqdm
from tqdm.notebook import tqdm 
import time
import random
import pickle
```

## Usage
### EmotionClassifier Object
The EmotionClassifier object is an object-oriented class created to hold all the information about our system. The subfunctions that exsist are: 
- init: initalization function for obj
  - Parameters: 
    - train_file_name (csv file for train)
    - test_file_name (usally empty, but filename if recieved)
    - test_file = false (simple boolean statement to see if test file)
    - exclude_neutral (parameter to generate train based on neutrals, defaults to false)
  - Function/Output: Init will preprocess the sets to then be ready to call train function
    
```python
    ec = EmotionClassifier('train.csv', 'test.csv', exclude_neutral=False)
```

- pretty_print: Function to print nicely for debug and padding consistency
  - type info: type of info presentedc
  - location: where in code
  - information: text to print

```python
pretty_print('DONE', 'VALIDATE_TRAIN', 'Validation Complete!')
pretty_print('START', 'OVERSAMPLER', 'Starting Oversampling...')
pretty_print('START', 'TEST_MODEL', 'Starting Test')
```

- preprocess_test_corpus: function to take in csv file name, and output preprocessed dataframe
  - Parameters:
    -  test_filename : filename of test set
    -  exclude_neutral : parameter to include neutral in test set or not
  - Function/Output: will set test_corpus variable to preprocessed corpus

```python
ec = EmotionClassifier(...)
ec.preprocess_test_corpus('test.csv') 
```

- test_model: function to take generated model and use it to predict on test
  - Parameters:
    - self: uses object variables declared earlier
  - Function/Output: iterates thrugh test corpus and tries to predict each based on model.... will pass out resutls including accuracy, macro acc, recall, precision, f1

```python
ec = EmotionClassifier(...)
ec.test_model()

```
- print_training_stats: output function to print results from saved parameters
  - Parameters:
    - self: uses object variables declared earlier
  - Function/Output: will output info on sets before and after, and validation using 20% of train set as test set

```
ec = EmotionClassifier(...)
# ec.train_model(...) already run, ec.validate_train() already run
ec.print_training_stats()
```

- oversample: function to oversample a df given a df (balancing)
  - Parameters:
    - df: dataframe read from csv, will be preprocessed
- Function/Output: outputs a dataframe upsampled

```
# split of x_train, y_train
#... (vectorization of text and label into X and y respectivly)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
data = {'text':X_train, 'label': y_train}
oversampled = self.oversample(pd.DataFrame(data))
X_train = oversampled['text'] 
y_train = oversampled['label']
```

- [NOT USED IN FINAL INFO PRINTED] generate_prediction_classreport : function to generate classreport from metrics library
  - Parameters:
    - features : features generated in train func, pandas series
    - labels : labels generated in train func, pandas series
    - df_preprocessed_corpus : entire preprocessed corpus
  - Function/Output: will set model_classreport_train var to class report -> FUNCTION NOT REALLY NEEDED IN END IMPLEMENTATION info on training accuracy

```python
# ...
#features = tfidf.fit_transform(...
#labels = df_preprocessed_corpus..
# pre = preprocess(...
self.generate_prediction_classreport(features, labels, pre)


```
- [NOT USED IN FINAL INFO PRINTED] generate_acc_wcv: function to use cross validation (5 fold) to generate accuracy
  - Parameters:
    - model: model being used for stats
    - features : features of model
    - labels : labels of model
  - Function/Output: outputs acc, which is accuracy dataframe containing cross_val_score -> not really used in end emp

```python
# ...
self.model_acc_train = self.generate_acc_wcv(LinearSVC(), features, labels)
```

- train_model: main function to train and generate model using linear SVC given a preprocessed corpus
  - Parameters:
      - df_preprocessed_corpus : corpus with proeprocessing done by preprocess func
    - Function/Output: function will vectorize with tfidf to get features and labels, will then split into train and test (which is really validate), function then oversamples using oversample func, function will then generate some metrics and then finally create comppete model using tfidf vectorizer and fitting it

```python
#features = tfidf.fit_transform(...
#labels = df_preprocessed_corpus..
# pre = preprocess(...
self.model_acc_train = self.generate_acc_wcv(LinearSVC(), features, labels)

```
- predict_emotion: helper function to predict given trained model and a text
  - Parameters:
    - text: a text that needs to be predicted
  - Function/Output: will output a result in the fourm of array of single emotion val

```python
ec = EmotionClassifier(ec)
# init, train model, etc 
ec.predict_emotion('i am angry')
```


- validate_train: function to validate training model based on using proportion of train to test, very similar to the test_model function
  - Parameters:
    - self: parameters generated from other functions, calling own vars using self
  - Function/Output: function will iterate through each text and emotion in validation set and then predict based on line given, and append to all results. Metrics are then simply generated by counting if predicted label matches given golden label and are set to object vars using self call

```python
ec = EmotionClassifier(ec)
# init, train model, etc 
ec.validate_train()
```

- preprocess: MAIN preprocessing function runner, will loop through each line of corpus and preprocess
  - Parameters:
    - corpus_df: entire corpus not preprocessed
    - exclude_neutral: parameter to preprocess and include or exclude neutral labels
  - Function/Output: calls subfunction preprocess_each_line which preforms actual preprocessing, more of runner func

```python
pre = self.preprocess(self.train_corpus, exclude_neutral=True)
#...
```

- preprocess_each_line : main preprocessing subfunction, will take a line and output preprocessed version
    - Parameters:
      - corpus_df_text_line: line in corpus to be preprocessed
      - corpus_df: entire corpus df read from pandas, used to get size and when removing interally in this func (now external)
    - Function/Output: function preprocesses by: case folding, removing numerals, removing [name] token entirely, remove any punctuation, tokenize with TweetTokenizer, filter out stop words, lemantize each with wordnet, and remove left over punc chars


```python
df = pandas.read_csv('train.csv')
text = 'I am hungry! wow :)'
pre = self.preprocess_each_line(text, corpus_df):
#...
```


### Using the EmotionClassifier Object
You can create a version of the EmotionClassifier with neutral labels
```python
# (  TASK 1.1  )
# create classifier object
ec = EmotionClassifier('T1_train.csv', 'NONE')
# train model based on preprocessed corpus generated in init
ec.train_model(ec.train_corpus_preproccessed)
# Validate results using 20% of train set as fake test set (validation set)
ec.validate_train()
# Print training stats, including from validation 
ec.print_training_stats()
# you can now preprocess the test corpus
ec.preprocess_test_corpus('T1_testFAKE.csv') 
# and test the model (which prints its own stats)
ec.test_model()

```
or without neutral labels 

```python
ec_noneutral = EmotionClassifier('T1_train.csv', 'NONE', exclude_neutral=True)
#...
ec_noneutral.preprocess_test_corpus('T1_testFAKE.csv', exclude_neutral=True)
ec_noneutral.test_model()
```

you can also pickle the object using a helper function like pickle_object_save and load it later on:

```python3
def pickle_object_save(object, filename):
    # Function to save entire classifier object as pickle file
  with open(filename, 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)

#...
pickle_object_save(ec, 'EmotionClassifierObject.pkl')

ec_frompickle = None
with open('EmotionClassifierObject.pkl', 'rb') as inp:
  ec_frompickle = pickle.load(inp)

print(ec.predict_emotion('I love some flowers'))

```
## Contributing
File is saved on RC enviorment and will be uploaded to github after test data released. 

## Sources for Creation of System
https://www.analyticsvidhya.com/blog/2021/11/a-guide-to-building-an-end-to-end-multiclass-text-classification-model/
https://stackoverflow.com/questions/34714162preventing-splitting-at-apostrophies-when-tokenizing-words-using-nltk
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-multilabel-svm-classifier-with-scikit-learn.md
https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
https://medium.com/analytics-vidhya/undersampling-and-oversampling-an-old-and-a-new-approach-4f984a0e8392


