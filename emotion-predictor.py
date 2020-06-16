import re
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
nltk.download('punkt')
nltk.download('stopwords')
from keras.models import load_model

import sys

phrase = ""
file = ""
isSinglePrediction = False

if sys.argv[1] == "single":
  isSinglePrediction = True
  phrase = ' '.join(sys.argv[2:])
else:
  file = sys.argv[2]
  
def lower_token(tokens): 
        return [w.lower() for w in tokens] 
    
def remove_stop_words(tokens, stoplist): 
        return [word for word in tokens if word not in stoplist]
    
def remove_punct(text):
        text_nopunct = ''
        text_nopunct = re.sub('['+string.punctuation+']', '', text)
        return text_nopunct

def predictEmotion(phrase, file, isSinglePrediction):
    
    if isSinglePrediction:
        list1 = [phrase]
        df = pd.DataFrame(list1)
        file = 'C:\\Intel\\Emotion_Predictor\\emotion_predictor.csv'
        df.to_csv(file, index=False, header=False)
    
    data = pd.read_csv(file, 
                   header = None, 
                   delimiter=',')
        
    data.columns = ['Text']
    
    data['Text_Clean'] = data['Text'].apply(lambda x: remove_punct(x))
    
    # Tokenizing the words
    tokens = [word_tokenize(sen) for sen in data.Text_Clean]   
       
    # Lower casing the words
    lower_tokens = [lower_token(token) for token in tokens]
    
    # Removing the stop words
    stoplist = stopwords.words('english')
    filtered_words = [remove_stop_words(sen, stoplist) for sen in lower_tokens]
    
    result = [' '.join(sen) for sen in filtered_words]
    
    data['Text_Final'] = result
    data['Tokens'] = filtered_words
    data = data[['Text_Final', 'Tokens']]
    
    all_test_words = [word for tokens in data["Tokens"] for word in tokens]
    TEST_VOCAB = sorted(list(set(all_test_words)))
    
    # Tokenizing and Padding
    MAX_SEQUENCE_LENGTH = 50
    
    tokenizer = Tokenizer(num_words=len(TEST_VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(data["Text_Final"].tolist())
    
    test_sequences = tokenizer.texts_to_sequences(data["Text_Final"].tolist())
    test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    # Loading the CNN
    model = load_model("C:\\Intel\\Emotion_Predictor\\cnn_model")
    
    # Predicting using the CNN
    predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)

    
    labels = ['happiness', 'sadness', 'anger', 'surprise', 'fear']
    
    happiness = 0;
    sadness = 0;
    anger = 0;
    surprise = 0;
    fear = 0;

    for p in predictions:
        predictedLabel = labels[np.argmax(p)];
        if (predictedLabel == "happiness"):
            happiness = happiness + 1;
        if (predictedLabel == "sadness"):
            sadness = sadness + 1;
        if (predictedLabel == "anger"):
            anger = anger + 1;
        if (predictedLabel == "surprise"):
            surprise = surprise + 1;
        if (predictedLabel == "fear"):
            fear = fear + 1;
    
    predictionCounts = [happiness, sadness, anger, surprise, fear]
    print(predictionCounts)
    
    if isSinglePrediction:
        prediction_labels=[]
        for p in predictions:
            prediction_labels.append(labels[np.argmax(p)])
    
        print(predictions[0])
        print(prediction_labels[0])
   

predictEmotion(phrase, file, isSinglePrediction)