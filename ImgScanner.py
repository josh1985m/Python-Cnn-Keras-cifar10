from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import cifar10
from tensorflow import keras
import numpy
import matplotlib.pyplot as plt

#seed = 21
#numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

#model.add(Conv2D(96, (3, 3), padding='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#keras.layers.GaussianNoise(1)
#model.add(Dropout(0.3))
#model.add(BatchNormalization())

#model.add(Conv2D(96, (3, 3), padding='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
#model.add(BatchNormalization())

#model.add(Conv2D(128, (4, 4), padding='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(.1))
#model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.3))

#model.add(Dense(96))
#model.add(Activation('relu'))
#model.add(Dropout(.3))
#model.add(BatchNormalization())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(.4))
model.add(BatchNormalization())

#model.add(Dense(32))
#model.add(Activation('relu'))
#model.add(Dropout(0.3))
#model.add(BatchNormalization())

model.add(Dense(num_classes))
model.add(Activation('hard_sigmoid'))
#softmax
#exponential
#elu
#selu
#relu
#softplus
#softsign
#tanh
#sigmoid
#hard_sigmoid
#linear

epochs = 1

optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
#optimizer = keras.optimizers.Adagrad(learning_rate=.01)
#optimizer = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
#optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
#optimizer = keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
#optimizer = keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)




model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#mean_squared_error
#mean_absolute_error
#mean_absolute_percentage_error
#mean_squared_logarithmic_error
#squared_hinge
#hinge
#categorical_hinge
#logcosh
#huber_loss
#categorical_crossentropy
#sparse_categorical_crossentropy
#binary_crossentropy
#kullback_leibler_divergence
#poisson
#cosine_proximity
#is_categorical_crossentropy

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#prediction = model.predict_generator()