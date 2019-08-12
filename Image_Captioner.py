# -*- coding: utf-8 -*-


import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt

import string
import os
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import cv2
import tensorflow as tf
import tqdm as tqdm
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer


train=pd.read_csv('../input/flickr-image-dataset/flickr30k_images/flickr30k_images/results.csv', delimiter='|')
path="../input/flickr-image-dataset/flickr30k_images/flickr30k_images/flickr30k_images/"

train.columns=['image_name','comment_number','comment']
train['comment_number'].astype=np.int64
train['comment'].astype=str
ser=pd.Series(train['comment_number'], dtype=np.int64)


model = InceptionV3(weights='imagenet')
# Remove the last layer (output softmax layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)


def read(imgspath):
    img = cv2.imread(imgspath)
    #im1=cv2.imread(imgpath)
    #im2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #im3 = cv2.GaussianBlur(img, (5, 5), 0)
    im4 = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
    #im5 = cv2.dilate(im4, (3, 3))
    im6 = np.expand_dims(im4, axis=0)
    # preprocess the images using preprocess_input() from inception module
    im7 = preprocess_input(im6)
    #imgarr=np.array(im4, dtype=np.int64)
    #x = np.expand_dims(imgarr, axis=0)
    # preprocess images using preprocess_input() from inception module
    #x = preprocess_input(im4)
    #fea_vec = model_new.predict(x) # Get the encoding vector for the image
    #fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    # reshape from (1, 2048) to (2048, )
    #x = np.reshape(x,2048)
    return im7
    

def encode(imge):
    image = read(os.path.join(path,imge)) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

testimpath=('../input/qwerty/488B8997.jpg')
testimg = cv2.imread(testimpath)
testim4 = cv2.resize(testimg, (299, 299), interpolation=cv2.INTER_AREA)
    #im5 = cv2.dilate(im4, (3, 3))
testim6 = np.expand_dims(testim4, axis=0)
    # preprocess the images using preprocess_input() from inception module
testim7 = preprocess_input(testim6)
testfea=model_new.predict(testim7)


desc2 = {}

for i in range(1001):
    if (i+1)%5==0:
        desc2[train.ix[i,'image_name']] = encode(train.loc[i,'image_name'])
        

desc1={}
list0=[]

for i in range(1001):
    if  (i+1)%5==0:
        list0.append(train.ix[i,'comment'])
        #descriptions.append(list1)
        desc1[train.ix[i,'image_name']]=list0
        list0=[]
    else:
        list0.append(train.ix[i,'comment'])
    


for key in desc1:
    desc1[key]='startseq '+"".join(desc1[key])+' endseq'
    


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        all_desc.append(descriptions[key])
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d) for d in lines)



t = create_tokenizer(desc1)

vocab_size = len(t.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


max_length = 34


def create_sequences(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()
    
    # loop for ever over images
    for key in descriptions:
            # retrieve the photo feature
        #ppop=desc_list.split(' ')
            
        #for desc in desc_list.split('\n'):
        photo = photos[key]
                # encode the sequence
            #seq = [wordtoix[word] for word in ppop if word in wordtoix]
        seq = tokenizer.texts_to_sequences([descriptions[key]])[0]
                # split one sequence into multiple X, y pairs
        for i in range(1, len(seq)):
                    # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
                  # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                  # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=1645)[0]
                    # store
                    
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)

    
    return array(X1), array(X2), array(y)
    
                    

def data_generator(descriptions, photos, tokenizer, max_length):
# loop for ever over images

    while 1:
    #for key, desc_list in descriptions.items():
            # retrieve the photo feature
    
        X1train, X2train, ytrain= create_sequences(tokenizer, max_length, descriptions, photos)
        yield [[X1train, X2train], ytrain]
        


def h(vocab_size,max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    x1 = Dropout(0.25)(fe2)
    x2 = Dense(512, activation='relu')(x1)
    x3 = Dropout(0.15)(x2)
    fe2 = Dense(256, activation='relu')(x3)
    #fe2 = Dense(256, activation='relu')(fe1)
    #fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(34,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    y1 = Dense(1024, activation='relu')(decoder1)
    y2 = Dense(1024, activation='relu')(y1)
    decoder2 = Dense(256, activation='relu')(y2)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    #outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    #model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model


model=h(vocab_size,34)
model.summary()


epochs = 1
steps = len(desc1)
for i in range(epochs):
    # create the data generator
    generator = data_generator(desc1, desc2, t, 34)
    # fit for one epoch
    model.fit_generator(generator, epochs=5, steps_per_epoch=steps, verbose=1)
    # save model
    #model.save('model_' + str(i) + '.h5')
    

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, t, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        seq = t.texts_to_sequences([in_text])[0]
        #seq = [wordtoix[word] for word in ppop if word in wordtoix]
        # pad input
        sequence = pad_sequences([seq], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat,t)
        in_text +=' ' + word    # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            in_text+=' '+word
            break
    return in_text
 
    
        #in_text += ' ' + word
        #if word == 'endseq':
        #    break
    #final = in_text.split()
    #final = final[1:-1]
    #final = ' '.join(final)
#return final
# load the tokenizer
#tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
#model = load_model('model-ep002-loss3.245-val_loss3.612.h5')
# load and prepare the photograph
#photo = array(testfea)
# generate description
description = generate_desc(model, t, testfea, 34)
print(description)


