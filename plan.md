# Austen Detector

**Goal**: Given a paragraph, detect if it is writen by Jane Austen or not.

Binary classification & NLP problem - text classifier

Quick check of given training dataset size:
~~~bash
~/recruitment-ml$ cat data/gutenberg-paragraphs.json | grep "\"austen\": 1" | wc -l
4749
~/recruitment-ml$ cat data/gutenberg-paragraphs.json | grep "\"austen\": 0" | wc -l
7861
~~~

## Approaches and Considerations

There are several possible viable model types to choose from, and in these there are further considerations to make on text encoding.

### Models:
- **Classical models**: Less likely to perform as well but quick to set up and run for analysis.
- **SVM**
- **Naive Bayes**
- **Deep NN**: Better chance at understanding the connection of words
- **Fully Connected NN**: Depending on text encoding / model architecture it won't have context for sentece structure - just vocabulary
- **CNN**: Expansion of above with better understanding of relational importance
- **LSTM**: Good for sequential data
- **Transformer**: For testing purpose use pre-trained model with fine-tuning / few-shot training

### Text Encoding

Possible encodings:
- **BoW**: Bag of Words, simple to implement. Lacks relational context
- **Word Embeddings**: Capture semantic meanings - pretrained embeddings can be used such as FastText
- **Tokenisation/1-hot encoding**: Store sentences as sequences of words/tokens. Required for models such as LSTMs or transformers

### Evaluation

As this is a binary classificaiton problem we can consider the confusion matrix and associated metrics e.g. accuracy, precision, recall, F1-score, and AUC-ROC.

We can further plot Precision-recall and ROC graph

Primary metrics to optimise for will be accuracy and AUC-ROC. Trade off between precision and recall is hard to tell without further productionisation knowledge - what is the cost of a False-postive?

Cross-Validation will also be needed to test the over all robustness of the models. The dataset size is small and this problem can likely be sclaed to other tasks, therefore we want better confidence in the model to perform the task more generically rather than just for the data we have for this exercise.

A further consideration to make is cost computationally financially. The dataset size is small, so running the test models should be low cost - but training times will be recorded for comparison.

### Hyperparameter Tuning

I will perform some tuning, but will keep it limited in the interest of time - this is likely one area for further exploration.


