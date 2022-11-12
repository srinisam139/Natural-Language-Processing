**FOR ENGL 581 (Natural Language Processing)**
Fall 2022     |     Kenyon, Vrushabh, Ketaki, Shrinivas

**Environment: # Make sure to run this with 5 cores and the following environment args: env activate engl-581-2**

## General_Structrure:-
- There are two python files topic_classifier.py and PS2-2_test.ipynb
- topic_classifier.py contains the code for preprocessing, vectorization and modeling
- PS2-2_test.ipynb contains the standalone python file used for testing
## Training:-
- Open the PS2-2_test.ipynb
- topic_classifier.py is already imported
- For the below function there are three arguments
    1. String - train/test
    2. filename - train/test data.csv
    3. model_file - pickled_file
- TopicClassifier("train/test", "train/testdata.csv", "model_file")
- Initally TopicClassifier argument will be passed in the following manner for **training**
    1.TopicClassifier("train", "train.csv", "LDA_model_params_8")
    **Note**:
         - During Training the model_file doesn't exist becasue the model is not yet saved
## Testing:-
- Open the PS2-2_test.ipynb
- topic_classifier.py is already imported
- For the below function there are three arguments
    1. String - train/test
    2. filename - train/test data.csv
    3. model_file - pickled_file
- TopicClassifier("train/test", "train/testdata.csv", "model_file")
- Initally TopicClassifier argument will be passed in the following manner for **testing**
    1.TopicClassifier("test", "test.csv", "LDA_model_params_8")
    **Note**:
         - The following model_file named "LDA_model_params_8" is already saved in our package, you can test the the data without training