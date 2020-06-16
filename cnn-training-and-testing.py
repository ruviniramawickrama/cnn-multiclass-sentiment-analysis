import re
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from gensim import models
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.callbacks import EarlyStopping
nltk.download('punkt')
nltk.download('stopwords')

# Reading the csv file
data = pd.read_csv('C:\\Intel\\Emotion_Predictor\\training-and-testing-dataset.csv', 
                   header = None, 
                   delimiter=',')

# Modifying the table columns for the loaded data
data.columns = ['Text', 'Label']

print("Sentiment Labels ----------")
print(data.Label.unique())

happiness = []
sadness = []
surprise = []
anger = []
fear = []

for label in data.Label:
    if label == 'happiness':
        happiness.append(1)
        sadness.append(0)
        surprise.append(0)
        anger.append(0)
        fear.append(0)
    elif label == 'sadness':
        happiness.append(0)
        sadness.append(1)
        surprise.append(0)
        anger.append(0)
        fear.append(0)
    elif label == 'surprise':
        happiness.append(0)
        sadness.append(0)
        surprise.append(1)
        anger.append(0)
        fear.append(0)
    elif label == 'anger':
        happiness.append(0)
        sadness.append(0)
        surprise.append(0)
        anger.append(1)
        fear.append(0)
    elif label == 'fear':
        happiness.append(0)
        sadness.append(0)
        surprise.append(0)
        anger.append(0)
        fear.append(1)

data['happiness']= happiness
data['sadness']= sadness
data['surprise']= surprise
data['anger']= anger
data['fear']= fear

print("\nData after adding new columns ----------")
print(data.head())

# Removing the punctuation marks
def remove_punctutations(text):
    text_clean = ''
    text_clean = re.sub('['+string.punctuation+']', '', text)
    return text_clean

data['Text_Clean'] = data['Text'].apply(lambda x: remove_punctutations(x))

# Tokenizing the words
tokens = [word_tokenize(sentence) for sentence in data.Text_Clean]

def lowercase_token(tokens): 
    return [word.lower() for word in tokens]    

# Lowercasing the tokens    
lowercased_tokens = [lowercase_token(token) for token in tokens]

# Removing the stop words
stoplist = stopwords.words('english')

def remove_stop_words(tokens): 
    return [word for word in tokens if word not in stoplist]

filtered_words = [remove_stop_words(word) for word in lowercased_tokens]

result = [' '.join(word) for word in filtered_words]

data['Text_Final'] = result
data['Tokens'] = filtered_words
data = data[['Text_Final', 'Tokens', 'Label', 'happiness', 'sadness', 'surprise', 'anger', 'fear']]

print("\nData after removing punctuation marks, stop words and lower casing ----------")
print(data.head())

# Splitting data into test and train
training_data, testing_data = train_test_split(data, test_size=0.10, random_state=42)

print("\nData after splitting into Train and Test sets ----------\n")

training_words = [word for tokens in training_data["Tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in training_data["Tokens"]]
training_vocabulary = sorted(list(set(training_words)))
print("%s total of Training words with a vocabulary size of %s" % (len(training_words), len(training_vocabulary)))
print("Max sentence length is %s" % max(training_sentence_lengths))

testing_words = [word for tokens in testing_data["Tokens"] for word in tokens]
testing_sentence_lengths = [len(tokens) for tokens in testing_data["Tokens"]]
testing_vocabulary = sorted(list(set(testing_words)))
print()
print("%s total of Testing words with a vocabulary size of %s" % (len(testing_words), len(testing_vocabulary)))
print("Max sentence length is %s" % max(testing_sentence_lengths))

# Loading Google News Word2Vec model
word2vec = models.KeyedVectors.load_word2vec_format('C:\\Intel\\Emotion_Predictor\\GoogleNews-vectors-negative300.bin.gz', binary=True)
print("\nCompleted loading Google Word2Vec ----------")

# Getting Embeddings
def get_average_word2vec(tokens, vector, generate_missing=False, k=300):
    if len(tokens)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[token] if token in vector else np.random.rand(k) for token in tokens]
    else:
        vectorized = [vector[token] if token in vector else np.zeros(k) for token in tokens]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['Tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

training_embeddings = get_word2vec_embeddings(word2vec, training_data, generate_missing=True)

# Tokenizing and Padding
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300

tokenizer = Tokenizer(num_words=len(training_vocabulary), lower=True, char_level=False)
tokenizer.fit_on_texts(training_data["Text_Final"].tolist())
training_sequences = tokenizer.texts_to_sequences(training_data["Text_Final"].tolist())
training_word_index = tokenizer.word_index

print('\nFound %s unique tokens.' % len(training_word_index))

training_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_embedding_weights = np.zeros((len(training_word_index)+1, EMBEDDING_DIM))
for word,index in training_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)

testing_sequences = tokenizer.texts_to_sequences(testing_data["Text_Final"].tolist())
testing_cnn_data = pad_sequences(testing_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Defining the CNN
def ConvolutionalNeuralNetwork(embeddings, max_sequence_length, num_of_words, embedding_dim, labels_index):
    
    embedding_layer = Embedding(num_of_words, embedding_dim, weights=[embeddings], input_length=max_sequence_length, trainable=False)
    
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    sliding_window_heights = [2,3,4,5,6]

    for sliding_window_height in sliding_window_heights:
        l_conv = Conv1D(filters=100, kernel_size=sliding_window_height, activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)  
   
    predictions = Dense(labels_index, activation='softmax')(x)

    model = Model(sequence_input, predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model


print("\nCNN defined successfully ----------")

# Training the CNN
print("\nTraining the CNN----------")

labels = ['happiness', 'sadness', 'anger', 'surprise', 'fear']

model = ConvolutionalNeuralNetwork(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(training_word_index)+1, EMBEDDING_DIM, 
               len(list(labels)))

y_train = training_data[labels].values
x_train = training_cnn_data
y_tr = y_train

num_epochs = 100
batch_size = 32

es = EarlyStopping(monitor='val_loss', mode='min',verbose=1,)
  
hist = model.fit(x_train, y_tr, epochs=num_epochs, validation_split=0.1, shuffle=True, batch_size=batch_size, callbacks=[es])

print("\nCNN trained successfully ----------")

# Saving the CNN
model.save("C:\\Intel\\Emotion_Predictor\\cnn_model")
print("\nCNN saved successfully ----------")

# Testing the CNN
predictions = model.predict(testing_cnn_data, batch_size=1024, verbose=1)

labels = ['happiness', 'sadness', 'anger', 'surprise', 'fear']

prediction_labels = []

for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])

happiness = { "actualCount": 0, "happiness": 0, "sadness": 0, "anger": 0, "surprise": 0, "fear": 0}
sadness = { "actualCount": 0, "happiness": 0, "sadness": 0, "anger": 0, "surprise": 0, "fear": 0}
anger = { "actualCount": 0, "happiness": 0, "sadness": 0, "anger": 0, "surprise": 0, "fear": 0}
surprise = { "actualCount": 0, "happiness": 0, "sadness": 0, "anger": 0, "surprise": 0, "fear": 0}
fear = { "actualCount": 0, "happiness": 0, "sadness": 0, "anger": 0, "surprise": 0, "fear": 0}


def calculateCounts(sentimentObject, predictedSentiment):
    sentimentObject["actualCount"] = sentimentObject["actualCount"] + 1;
    if (predictedSentiment == "happiness"):
        sentimentObject["happiness"] = sentimentObject["happiness"] + 1;
    if (predictedSentiment == "sadness"):
         sentimentObject["sadness"] = sentimentObject["sadness"] + 1;
    if (predictedSentiment == "anger"):
         sentimentObject["anger"] = sentimentObject["anger"] + 1;
    if (predictedSentiment == "surprise"):
         sentimentObject["surprise"] = sentimentObject["surprise"] + 1;
    if (predictedSentiment == "fear"):
         sentimentObject["fear"] = sentimentObject["fear"] + 1;

i = 0    
for index, row in testing_data.iterrows():
       actualSentiment = row['Label']
       predictedSentiment = prediction_labels[i]       
       if (actualSentiment == "happiness"):
           calculateCounts(happiness, predictedSentiment)
       if (actualSentiment == "sadness"):
           calculateCounts(sadness, predictedSentiment)    
       if (actualSentiment == "anger"):
           calculateCounts(anger, predictedSentiment)           
       if (actualSentiment == "surprise"):
           calculateCounts(surprise, predictedSentiment)    
       if (actualSentiment == "fear"):
           calculateCounts(fear, predictedSentiment)
       i = i + 1

total_predictions = len(prediction_labels)
correct_predictions = sum(testing_data.Label==prediction_labels)
accuracy = (correct_predictions/total_predictions)*100

print("\ntotal_predictions")
print(total_predictions)

print("\ncorrect_predictions")
print(correct_predictions)    

print("\naccuracy")
print(accuracy)

print("\nHappiness counts")
print(happiness)

print("\nSadness counts")
print(sadness)

print("\nAnger counts")
print(anger)

print("\nSurprise counts")
print(surprise)

print("\nFear counts")
print(fear)


