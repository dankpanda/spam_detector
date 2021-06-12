import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

oov = "<OOV>"
numWords = 5000
paddingType = "post"
truncateType = "post"
maxLength = 100

model = tf.keras.models.load_model('my_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

while True:
    testA = input(">")
    test = []
    test.append(testA)
    sequences = tokenizer.texts_to_sequences(test)
    padded = pad_sequences(sequences, maxlen = maxLength, padding=paddingType)
    pred = model.predict(padded)
    labels = ['Normal SMS','Spam','Promo','OTP']
    print(labels[np.argmax(pred)])
    print(pred)




