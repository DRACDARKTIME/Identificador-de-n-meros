import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.regularizers import l1,l2,l1_l2 
from keras.callbacks import ModelCheckpoint, EarlyStopping #Guarda la mejor red
import matplotlib.pyplot as plt
import mlflow
#------------------------------------Parámetros-----------------------------------------------
learning_rate = 0.001
epochs = 30
batch_size = 10
beta_1 = 0.09
beta_2 = 0.9999
epsilon= 1e-7
modelo = 'Adam'
lambda2 = 0.001
#------------------------------------Datos del modelo-----------------------------------------
#Activa el servidor
mlflow.tensorflow.autolog()    
#Cargamos los datos
dataset  = mnist.load_data()   
(x_train, y_train), (x_test, y_test) = dataset
#Aplanamos a las imagenes
x_trainv = x_train.reshape(60000, 784).astype('float32') 
x_testv  = x_test.reshape(10000, 784).astype('float32')
#Normalizamos los datos
x_trainv /= 255 
x_testv /= 255 
#One-hot encoding
num_classes=10 
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)
#------------------------------------Modelo-----------------------------------------
model = Sequential()
#Regularizadores
kernel_regularizer = l2(lambda2)
#Capas
model.add(Dense(200, activation='sigmoid', input_shape=(784,),kernel_regularizer=kernel_regularizer))
#model.add(Dropout(0.2))
model.add(Dense(100, activation='relu',kernel_regularizer=kernel_regularizer))
#model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()
####Checkpoint and EarlyStopping#####
#specify path to save model
filepath = "best_model.hdf5" 
#Checkpoint to save the model
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') #el 'min' de 'val_loss'
#Early stopping
earlystop = EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=True,patience=10,verbose=1)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        amsgrad=True 
    ),
    metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc),
                    callbacks=[checkpoint,earlystop]
                    )
#------------------Gráficas---------------------------------
fig, ax=plt.subplots(1,2, figsize=(20,10))
ax[0].set_title(f'loss vs epoch,beta_1={beta_1},beta_2={beta_2},eta={learning_rate}')
ax[0].plot(history.history['loss'],'r', label='loss')
ax[0].plot(history.history['val_loss'],'b', label='val_loss')
ax[0].set_ylabel('loss')
ax[0].set_xlabel('epoch')
ax[0].legend()
ax[1].set_title('accuracy vs epoch')
ax[1].plot(history.history['accuracy'],'r', label='accuracy')
ax[1].plot(history.history['val_accuracy'],'b', label='val_accuracy')
ax[1].set_ylabel('loss')
ax[1].set_xlabel('epoch')
ax[1].legend()
fig.tight_layout()
plt.savefig(f"Modelo:{modelo}, l2, lambda2={lambda2}.jpg")
plt.show()
#en consola: mlflow ui
