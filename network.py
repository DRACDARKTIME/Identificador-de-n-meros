"""
network.py
~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#-------- Libraries --------#

import random
    # Standard library

import numpy as np
    # Third-party libraries

#-------- Definición de la Red Neuronal  --------#

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.
        Note that the first layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes) #Numero de capas
        self.sizes = sizes #Creamos un atributo 'sizes' a self 
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
            #Creamos un atributo 'biases' a self, donde asignamos una lista
            #Esta lista tiene como elementos a arrays, cada array pertenece a una capa de la red
            #Cada array tiene una 'b' random para cada neurona   
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
            #Ahora asignamos un peso w aleatorio entre cada par de neuronas 
            #Se crea un primer array con los pesos que unen a cada neurona entre la capa uno y dos
                #En este array hay otros array con n valores, esos valores corresponden a los w's
                #que tiene cada nuerona de la segunda capa a la primer capa. 
            #Se crea un segundo array con los pesos que unen a cada neurona entre la capa dos y tres
                #El ciclo se repite   
                #         .
                #         .
                #         .
                #Hasta que se alcanza a la última capa.

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights): #Tomamos un valor de self.biases y self.weights (son arrays)
            a = sigmoid(np.dot(w, a)+b) #Hace el producto punto entre w y a y lo evalua en la sigmoide.
                                        #Aquí debería de haber un problema con a, ya que w no tiene las mismas 
                                        #Dimensiones siempre entonces el producto debe fallar
        return a #Se crea un array con valores de la sigmoide para cada w

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):  #Stochastic Gradient Descent
                              #self            --- Llamamos a nuestra clase 'self'
                              #training_data   --- Una lista de tuplas (x,y) donde ('espectativa','realidad') xd  
                              #epochs          --- Ciclos a repetir
                              #mini_batch_size --- Es el paso que hay en la selección de los datos "step"
                              #eta             --- Ritmo de aprendizaje
                              #test_data       --- Información de prueba  
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            test_data = list(test_data) #La convertimos en lista
            n_test = len(test_data)     #Obtenemos su longitud

        training_data = list(training_data) #La convertimos en lista
        n = len(training_data)              #Obtenemos su longitud
        for j in range(epochs):
            random.shuffle(training_data)   #Revolvemos el orden de los datos en training_data

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] #Se crean los subconjuntos de training_data
                                                       #De tamaño = n/mini_batch_size
                                                       #mini_batch_size es el step de 0 a n

            for mini_batch in mini_batches:             #mini_batch es un dato de mini_batches
                self.update_mini_batch(mini_batch, eta) #Llamamos a la función update_mini_batch (definida abajo)
                                                        #Donde guardamos (mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)) #Imprimimos el número de época
                                                          #llama el valor de evaluate(función definida abajo)
                                                          #longitud de test_data
            else:
                print("Epoch {0} complete".format(j))     #Solo se imprime la epoca

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]     #Crea una lista igual a self.biases pero con puros ceros
        nabla_w = [np.zeros(w.shape) for w in self.weights]    #Crea una lista igual a self.weights pero con puros ceros
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)            #La función backprop(x,y) regresa dos valores
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #Ahora nabla_b está llena de la suma de los valores
                                                                          #nabla_b + delta_nabla_b
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #Ahora nabla_w está llena de la suma de los valores
                                                                          #nabla_w + delta_nabla_w
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]          #Actualizamos los pesos, moviendo un poco los w's
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]            #Actualizamos los biases, moviendo un poco los b's

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]       #Crea una lista igual a self.biases pero con puros ceros
        nabla_w = [np.zeros(w.shape) for w in self.weights]      #Crea una lista igual a self.weights pero con puros ceros
        # feedforward
        activation = x                                           #Input
        activations = [x] # list to store all the activations, layer by layer  Valores de sigma
        zs = [] # list to store all the z vectors, layer by layer              
        for b, w in zip(self.biases, self.weights):     
            z = np.dot(w, activation)+b                   
            zs.append(z)                      #Añadimos elementos a zs
            activation = sigmoid(z)
            activations.append(activation)    #Añadimos elementos a activations

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])  #Calculamos la primer delta
        nabla_b[-1] = delta                   #El último dato de nabla_b lo cambiamos por delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #Cambiamos el último dato de w, usando la nueva b

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)    #Hemos actualizado los valores de nabla_b y nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]    #Tomamos los máximos de las activaciones
        return sum(int(x == y) for (x, y) in test_results) #Vemos los resultados que coincidieron y los sumamos

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)  #Esta es la derivada de C_x respecto de $a^{L}$

#------------ Miscellaneous functions ------------#

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
