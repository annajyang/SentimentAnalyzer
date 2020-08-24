# SentimentAnalyzer
Created by Anna Yang, Jenny Yang, and Ayush Raj for Ignition Hacks 2020

Submission is in `submission.csv`. `submissions.zip` is a compressed zip file of an older version of our submission.

Our final notebook is `finalnotebook.ipynb`. We pre-processed the data by cleaning punctuation, capital letters, lemmatizing, and both tokenizing and creating 2-word phrases for each tweet that we would use as features, as shown in `preprocessing.ipynb`. Then, we normalized the data and then implemented an ANN and Logistic Regression on the training data. We then created predictions for the `contestant_judgement.csv` file and put that in `submissions.zip`, as the file was too large for github. `model.ipynb` also contains code for the model and classification.
