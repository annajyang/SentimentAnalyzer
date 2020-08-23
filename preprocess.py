import pandas as pd
import io
import numpy as np
import csv
import nltk
import re
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv('training_data.csv')
judging = pd.read_csv('contestant_judgment.csv')

stop_words = set(stopwords.words('english'))
filtered_sent = []

for index, row in train.iterrows():
    tokenized_words = []
    # Remove @s
    train.at[index, 'Text'] = re.sub(r"@[A-z]+", " ", train.at[index, 'Text'])
    
    # Remove punct
    train.at[index, 'Text'] = re.sub(r"\W", " ", train.at[index, 'Text'])

    # Remove numbers
    train.at[index, 'Text'] = re.sub(r"\d+", "", train.at[index, 'Text'])

    # Remove spaces
    train.at[index, 'Text'] = re.sub(r"\s+", " ", train.at[index, 'Text'])
   
    # Make lowercase
    train.at[index, 'Text'] = str(train.at[index, 'Text']).lower()
    
    # Tokenize
    tokenized_words = nltk.word_tokenize(train.at[index, 'Text'])
    
    filtered_sent.append([])
    for w in tokenized_words:
        if w not in stop_words:
            filtered_sent[index].append(w)
            
lemmatizer = WordNetLemmatizer()
i = 0
for words in filtered_sent:
    newWords = []
    
    for word in words:
        newWords.append(lemmatizer.lemmatize(word, pos = 'v'))
     
    filtered_sent[i] = " ".join(newWords)
    i += 1

new_column = pd.DataFrame({'BagofWords': filtered_sent})
train = train.merge(new_column, right_index = True, left_index=True)

bagofwords = train.BagofWords.tolist()
sentiment = train.Sentiment.tolist()

cv = CountVectorizer(max_features=1000)
cv2 = CountVectorizer(max_features=1000, ngram_range=(2, 3))

x_1 = cv.fit_transform(train['BagofWords']).toarray()
x_2 = cv2.fit_transform(train['Text']).toarray()
y = train['Sentiment'].values

header_1 = cv.get_feature_names()
header_2 = cv2.get_feature_names()
output_1 = pd.DataFrame(x_1, columns = header_1)
output_2 = pd.DataFrame(x_2, columns = header_2)
output_1 = output_1.merge(output_2, right_index=True, left_index=True)
output_1['Sentiment'] = train['Sentiment']

output_1.head()
output_1.to_csv (r'vectorized.csv', index = True, header=True)