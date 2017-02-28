import cPickle
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Reshape, Flatten
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers import TimeDistributed

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.preprocessing.sequence import pad_sequences
from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical

def index_to_input(X, maxlen_sent, maxlen_doc):
    """
    transform the index-list based input to some data that can be fed into the text_CNN_GRU class
    :param X: [[word_index]]
    :return:
    """
    X = [pad_sequences(i, maxlen=maxlen_sent, padding='post') for i in X]
    X = pad_2Dsequences(X, maxlen=maxlen_doc)
    return X


def pad_2Dsequences(sequences, maxlen=None, dtype='int32', padding='post',
                    truncating='post', value=0.):
    "modify the keras.preprocessing.sequence to make it padding on doc level"
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    nb_dims = sequences[0].shape[1]

    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen, nb_dims)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

    return x


x, y = cPickle.load(open('HR_monarchy_data.p'))

#x = index_to_input(x, 30, 300)

maxlen_doc = 7
maxlen_sent = 50
filter_length = [3, 4, 5, 6]
nb_filters = 200
n_vocab = 10000,
embedding_dims = 300
hidden_gru = 64
n_classes = 5

n_vocab, embedding_dims = embedding_mat.shape

print "Building the model"
#graph model
inputs = Input(name='input', shape=(300, 300),
               dtype='float32')

#inputs = Input(name='input', shape=(7, 50),
#               dtype='int32')

#Model embedding layer, for word index-> word embedding transformation
#emb = Embedding(n_vocab, embedding_dims,
#                weights=[embedding_mat],
#                input_length=maxlen_sent * maxlen_doc,
#                name='embedding')(inputs)
#emb = Reshape((maxlen_doc, 1, maxlen_sent,
#               embedding_dims), name='reshape_5d')(inputs)
#define the different filters
#conv_layer = []
#for each_length in filter_length:
conv = TimeDistributed(Convolution1D(nb_filters / 3, 3, embedding_dims,
                                     border_mode='valid', activation='relu',),
                       input_shape=(300, 1
                                    300,
                                    embedding_dims),
                       name='conv_{}'.format(3))(emb)

conv = TimeDistributed(MaxPooling1D(pool_size=(int(maxlen_sent - 3 + 1), 1),
                                    border_mode='valid',
                                    name='pool_conv_{}'.format(3)
                                    ))(conv)
conv = TimeDistributed(Flatten(name='flatten_conv_{}'.format(3)))(conv)
#    conv_layer.append(conv)
#z = merge(conv_layer, mode='concat')
rnn = Dropout(0.5)(GRU(hidden_gru, name='gru_forward',
                       return_sequences=True)(conv))
rnn = Dropout(0.5)(GRU(hidden_gru, go_backwards=True,
                       name='gru_backward')(rnn))
a = Dense(n_classes, name='full_con', activation='softmax')(rnn)

model = Model(input=inputs, output=a)

model.compile('adam', loss='categorical_crossentropy')



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=42)

y_train = np.asarray(y_train, dtype='int32') - 1
y_train = to_categorical(y_train, nb_classes=5)
y_test = np.asarray(y_test, dtype='int32') - 1
y_test = to_categorical(y_test, nb_classes=5)


model.fit(X_train, y_train, batch_size=32, nb_epoch=2,
          validation_data=(X_test, y_test))
