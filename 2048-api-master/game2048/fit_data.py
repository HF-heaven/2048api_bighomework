import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D

BATCH_SIZE = 128
NUM_CLASSES = 4
NUM_EPOCHS = 3

class TrainKeras:
    
    model = None
    
    def fit(self,score):
        # download and load the data (split them between train and test sets)
        download = np.load('epoch%d_fit.npz'%score)
        #download = np.load('train_data_fit.npz')
        X = download['data']
        Y = download['label']
        print('X_shape:',np.shape(X))
        print('Y_shape:',np.shape(Y))
        
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
        #x_train, x_test, y_train, y_test = X,Y,X[::10], Y[::10]

        # expand the channel dimension     
        x_train = x_train.reshape(x_train.shape[0], 4, 4, 1)
        x_train = self.one_hot(x_train)
        x_test = x_test.reshape(x_test.shape[0], 4, 4, 1)
        x_test = self.one_hot(x_test)
        input_shape = (4, 4, 10, 1)

        # make the value of pixels from [0, 255] to [0, 1] for further process
        x_train = x_train.astype('float32') 
        x_test = x_test.astype('float32') 

        # convert class vectors to binary class matrics
        y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

        self.model = None
        self.model = Sequential()
        self.model.add(Conv3D(32,(3,3,3),padding='same'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        #self.model.add(Conv2D(16,(3,3),padding='same'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.35))
        self.model.add(Dense(NUM_CLASSES, activation='softmax'))
        # define the object function, optimizer and metrics
        self.model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
        

        # train
        self.model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_data=(x_test, y_test))
        
    def board_to_move(self,data):
        data = data.reshape(1,16)
        trans = self.one_hot(data)
        trans = trans.reshape(1,4,4,10,1)
        predict = self.model.predict_classes(trans)
        print(predict)
        return predict[0]
    
    def one_hot(self,data):
        n = data.shape[0]
        data = data.reshape(n, 16)
        one_hot_model = np.zeros([n,16,10])
        for k in range(n):
            for i in range(16):
                one_hot_model[k,i,int(data[k,i])-1] = 1
        one_hot_model = one_hot_model.reshape(n,4,4,10,1)
        return one_hot_model
                
'''
# evaluate
score_train = model.evaluate(x_train, y_train, verbose=0)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0], score_train[1]*100))
score_test = model.evaluate(x_test, y_test, verbose=0)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (score_test[0],score_test[1]*100))


Training loss: 0.2103, Training accuracy: 94.04%
Testing loss: 0.2097, Testing accuracy: 94.07%
'''