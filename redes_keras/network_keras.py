#https://www.tutorialspoint.com/keras/keras_installation.htm
#https://docs.python.org/es/3/tutorial/venv.html
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping #Para guardar la red y para detener la red cuando ya sobreajusta.
import matplotlib.pyplot as plt
learning_rate = 0.001
epochs = 30
batch_size = 120
###################
#pip install -q mlflow
import mlflow
#Servidor que guarda el historial de entrenamieto, guarda el modelo y los parámetros. Para ver el más chido.
mlflow.tensorflow.autolog() #Empezamos a activar el servidor
dataset=mnist.load_data() #Cargamos los datos

(x_train, y_train), (x_test, y_test) = dataset #Se separan por entrenamiento y prueba
#print(y_train.shape)
#print(x_train.shape)
#print(x_test.shape)
#x_train=x_train[0:8000]
#x_test=x_train[0:1000]
#y_train=y_train[0:8000]
#y_test=y_train[0:1000]
x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)
#print(x_trainv[3])
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')
x_trainv /= 255  # x_trainv = x_trainv/255
x_testv /= 255
#print("linea 40--------")
#print(x_trainv[3])
#print(x_train.shape)
#print(x_trainv.shape)
num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)
#print(y_trainc[6:15])
model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=(784,)))
#model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
#model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()
####Checkpoint and EarlyStopping#####
filepath = "best_model.hdf5" #specify path to save model
#Checkpoint to save the model
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #el 'min' de 'val_loss'
#Early stopping
earlystop = EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=True,patience=10,verbose=1) # verbose =
model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=learning_rate),
                metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc),
                    callbacks=[checkpoint,earlystop]
                    )
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
#en consola: mlflow ui