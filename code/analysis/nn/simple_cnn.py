'''
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python quad_classifier_v10.py
'''

from __future__ import print_function
import json
import numpy as np
import cPickle
from sklearn.cross_validation import train_test_split

np.random.seed(0123)  # for reproducibility

from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D


# set parameters:
maxlen = 300
batch_size = 32
embedding_dims = 300
nb_filter = 150
ngrams = [3, 4, 5]
nb_epoch = 3
subset = None


print('Loading data...')
x, y = cPickle.load(open('HR_monarchy_data.p'))


print('Load word vectors...')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

print('X_train shape:', x_train.shape)
print('X_test shape:', x_test.shape)

print('Build model...')


inputs = Input(shape=(300, 300,), name='input', dtype='float32')

#conv_filters = []
#for ngram in ngrams:
#    # we add a Convolution1D, which will learn nb_filter
#    # word group filters of size filter_length:
#    conv = Convolution1D(nb_filter=nb_filter, filter_length=ngram,
#                         border_mode='valid', activation='relu',
#                         subsample_length=1)(inputs)
#    # we use standard max pooling (halving the output of the previous layer):
#    conv = MaxPooling1D(pool_length=2)(conv)
#
#    # We flatten the output of the conv layer,
#    # so that we can add a vanilla dense layer:
#    conv = Flatten()(conv)
#    conv_filters.append(conv)
#
## Merge the conv layers together and add some dropout to the layer outputs
#z = Dropout(0.5)(merge(conv_filters, mode='concat'))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
conv = Convolution1D(nb_filter=nb_filter, filter_length=3,
                        border_mode='valid', activation='relu',
                        subsample_length=1)(inputs)
# we use standard max pooling (halving the output of the previous layer):
conv = MaxPooling1D(pool_length=2)(conv)

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
conv = Flatten()(conv)

z = Dropout(0.5)(Dense(200, activation='relu')(conv))
z = Dropout(0.5)(Dense(200, activation='relu')(z))
# Fully connected layer with ReLU activation
z = Dense(nb_filter, activation='relu')(z)
# Full connected output layer with softmax activation
z = Dense(1, activation='softmax', name='output')(z)

model = Model(input=inputs, output=z)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(x_test, y_test))

print('Saving model params...')
json_string = model.to_json()
with open('1d_cnn_model.json', 'w') as f:
    json.dump(json_string, f)

model.save_weights('1d_cnn_model_weights.h5')
