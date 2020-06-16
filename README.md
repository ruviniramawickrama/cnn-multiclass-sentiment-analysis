# Multi Class Sentiment Analysis using Convolutional Neural Networks (CNN)
This repository contains python scripts to perform multi class sentiment analysis. The supported sentiment classes are "Happiness", "Sadness", "Anger", "Surprise" and "Fear".

----------------------------------------------------------------------------------------------------------------------------------------
The file "training-and-testing-cnn.py" is used to train and test the CNN. It uses one hot encoding and word embeddings and trains the CNN. Then the trained model is tested and predictions are made. 

The prerequisite to run this file is listed below
  1. Create a folder named "Emotion_Predictor" in the location C:\Intel\
  2. Download and add the dataset "training-and-testing-dataset" to the location. The same dataset will be split and used for training        and testing. If you are planning to use a different dataset, include the dataset in the same location with the same name "training-      and-testing-dataset" and the dataset should be in csv format. There should be two columns in the dataset, first column is for the        phrases and the second column is for the sentiment label which should be labelled as "happiness", "sadness", "anger", "surprise"        and "fear"
  3. Download "GoogleNews-vectors-negative300.bin" and include it in the same location

Two major outputs are created by running this file.
  1. Trained cnn model will be saved to C:\Intel\ with the name "cnn_model"
  2. After testing the model, the script prints the actual and the predicted counts which can be used for further analysis.
     For example, happiness counts will be printed as: {'actualCount': 658, 'happiness': 463, 'sadness': 121, 'anger': 4, 'surprise':        66, 'fear': 4}
     This means out of all the test data, 658 are labelled as happinees. However model correctly predicts only 463 of them and the            others are preicted as "sadness", "anger" etc. 

----------------------------------------------------------------------------------------------------------------------------------------

The file "emotion-predictor.py" can be used to predict emotion of a single phrase or predict the emotions of multiple phrases and provide a consolidated output

The prerequisite to run this file is listed below
  1. Create a folder named "Emotion_Predictor" in the location C:\Intel\
  2. The trained CNN model "cnn_model" should be included in the above location (you can either use the cnn_model uploaded in the            repository or create it by running "training-and-testing-cnn.py")
  3. An empty CSV file with the name "emotion_predictor" should be created in the above location
  
To predict a single emotion, execute the below command in the command line
    python <location to emotion-predictor.py> "single" <phrase to predict the emotion>
    python C:\Intel\emotion-predictor.py "single" "Today is a great day!!!"

The output of the above command will be probablities for each sentiment class "Happiness", "Sadness", "Anger", "Surprise" and "Fear" respectively along with predicted sentiment class which has the highest probability.

To predict emotions of multiple phrases, execute the below command in the command line (The phrases should be included within a CSV file)
    python <location to emotion-predictor.py> "multiple" <location to CSV file having multiple phrases>
    python C:\Intel\emotion-predictor.py "multiple" "C:\\Intel\\Emotion_Predictor\\feedbacks.csv"

The output of the above command will be prediction counts for each sentiment class "Happiness", "Sadness", "Anger", "Surprise" and "Fear" respectively 
