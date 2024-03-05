import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD, Adam
from keras.regularizers import l1,l2,l1_l2 
from keras.callbacks import ModelCheckpoint, EarlyStopping #Guarda la mejor red
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
#------------------------------------Parámetros-----------------------------------------------
params = {
    'learning_rate' : 0.001,
    'epochs' : 3,
    'batch_size' : 10,
    'beta_1' : 0.09,
    'beta_2' : 0.9999,
    'epsilon': 1e-7,
    'lambda1' : 1e-5,
    'lambda2': 1e-5,
    'porcentaje':0.20,
    'modelo':'Adam'
}
#------------------------------------Datos del modelo-----------------------------------------
#Activa el servidor
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment('vamos a ver')
with mlflow.start_run():
    # mlflow.set_experiment('Digits_experiment')
    # with mlflow.start_run():
    #     mlflow.log_params(params)
    mlflow.log_params(params)   
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
    kernel_regularizer = l1_l2(params['lambda1'],params['lambda2'])
    #Capas
    model.add(Dense(200, activation='sigmoid', input_shape=(784,),kernel_regularizer=kernel_regularizer))
    model.add(Dropout(params['porcentaje']))
    model.add(Dense(100, activation='relu',kernel_regularizer=kernel_regularizer))
    model.add(Dropout(params['porcentaje']))
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
            learning_rate=params['learning_rate'],
            beta_1=params['beta_1'],
            beta_2=params['beta_2'],
            epsilon=params['epsilon'],
            amsgrad=True 
        ),
        metrics=['accuracy'])
    history = model.fit(x_trainv, y_trainc,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=1,
                        validation_data=(x_testv, y_testc),
                        callbacks=[checkpoint,earlystop]
                        )
    mlflow.keras.log_model(history, 'model')
#------------------Gráficas---------------------------------
fig, ax=plt.subplots(1,2, figsize=(20,10))
ax[0].set_title('loss vs epoch,beta_2:{0},eta:{1}'.format(params['beta_2'],params['learning_rate']))
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
plt.savefig("Modelo:{0}, Dropout:{1}, lambda1={2},lambda2={3}, epocas:{4}.jpg".format(
    params['modelo'],params['porcentaje'],params['lambda1'], params['lambda2'],params['epochs'])
)
plt.show()
#en consola: mlflow ui
