import numpy as np
import math
from sklearn.metrics import accuracy_score


class Numras:
    def __init__(self, layers, activation, input_size, optimizer=None):
        self.layers = layers
        self.L = len(layers)
        self.layers.insert(0, input_size)
        self.costs = list()
        self.history = {
            "test_acc" : list(),
            "loss" : list(),
            "train_acc" : list()
        }
        self.parms = dict()
        self.activation = activation
        # adding optimizer
        self.optimizer = optimizer
        if optimizer: self.optimizer.init_params(self.layers)
        print(self.layers)
        
    
    def init_params(self, method="normal"):
        for l in range(1, len(self.layers)):
            self.parms[f"W{l}"] = np.random.rand(self.layers[l], self.layers[l-1])
            self.parms[f"b{l}"] = np.zeros((self.layers[l], 1))
            if method == "normal":
                self.parms[f"W{l}"] = self.parms[f"W{l}"] * 0.01
            if method == "xavier":
                self.parms[f"W{l}"] = self.parms[f"W{l}"] * np.sqrt(2/(self.layers[l]+self.layers[l-1]))
            if method == "he":
                self.parms[f"W{l}"] = self.parms[f"W{l}"] * np.sqrt(2/(self.layers[l-1]))
                
              
    def forward(self, X):
        cache = dict()
        
        A = X
        
        for i in range(1, len(self.layers)):
            W = self.parms[f"W{i}"]
            b = self.parms[f"b{i}"]
            Z = np.dot(W, A) + b
            if self.activation[i-1] == "relu":
                A = self.relu(Z)
            if self.activation[i-1] == "softmax":
                A = self.softmax(Z)
            
            cache[f"A{i}"] = A
            cache[f"Z{i}"] = Z
            cache[f"W{i}"] = W
       # print(A[0, 0])
        return cache[f"A{self.L}"], cache
    
    def backward(self, X, Y, cache):
        derivatives = dict()
        cache[f"A0"] = X
        
        A = cache[f"A{self.L}"]
        A_prev = cache[f"A{self.L-1}"]
        
        # output layer 
        if self.activation[-1] == "softmax":
            dZ = A - Y
        dW = np.dot(dZ, A_prev.T) / dZ.shape[1]
        db = np.sum(dZ, axis=1, keepdims=True) / dZ.shape[1]
        dA_prev = np.dot(cache[f"W{self.L}"].T, dZ)
        
        
        derivatives[f"dW{self.L}"] = dW
        derivatives[f"db{self.L}"] = db
        
        for l in range(self.L-1, 0, -1):
            if self.activation[l-1] == "relu":
                dZ = dA_prev * self.relu_derivative(cache[f"Z{l}"])

            A_prev = cache[f"A{l-1}"]
            dW = np.dot(dZ, A_prev.T) / dZ.shape[1]
            db = np.sum(dZ, axis=1, keepdims=True) / dZ.shape[1]
            if l > 1:
                dA_prev = np.dot(cache[f"W{l}"].T, dZ)
            
            derivatives[f"dW{l}"] = dW
            derivatives[f"db{l}"] = db
            
        return derivatives
    
    def fit(self, X, Y, learning_rate=0.01, epoches=10, init_method="xavier", mini_batch_size=64, validation_set=None):
        if validation_set:
            test_x, test_y = validation_set
        
        self.init_params(method=init_method)
        
        
        for loop in range(epoches):
            np.random.seed(loop)
            mini_batches = self.random_mini_batches(X, Y, mini_batch_size = mini_batch_size)
            for c, mini_batch in enumerate(mini_batches):
                x, y = mini_batch
                A, cache = self.forward(x)
                
                derivatives = self.backward(x, y, cache)
                
                if self.optimizer:
                        derivatives = self.optimizer.take_step(derivatives)
                
                # updating weights
                for l in range(1, len(self.layers)):    
                    self.parms[f"W{l}"] = self.parms[f"W{l}"] - derivatives[f"dW{l}"]
                    self.parms[f"b{l}"] = self.parms[f"b{l}"] - derivatives[f"db{l}"]
                
            cost = - np.mean(y * np.log(A+1e-8))
            self.history["loss"].append(cost)
            
                
            if (loop+1) % 1 == 0:
                train_acc = self.accuracy(Y, self.predict(X))
                self.history["train_acc"].append(train_acc)
                if validation_set:
                    test_acc = self.accuracy(test_y, self.predict(test_x))
                    print(f"Epoch {loop+1} Cost : {cost} Train Accracy : {100*train_acc} Test Accuracy: {100*test_acc}")
                    self.history["test_acc"].append(test_acc)
                else: 
                    print(f"Epoch {loop+1} Cost : {cost} Train Accracy : {100*train_acc}")
            
            
    
    def predict(self, X):
        A, cache = self.forward(X)
        return A
    
    def accuracy(self, Y, Y_pred):
        y_hat = np.argmax(Y_pred, axis=0)
        Y = np.argmax(Y, axis=0)
        return accuracy_score(y_hat, Y)
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / expZ.sum(axis=0, keepdims=True)
    
    def relu_derivative(self, x):
        dZ = np.array(x, copy=True)
        dZ[x <= 0] = 0
        dZ[x > 0] = 1.0
        return dZ
    
    def random_mini_batches(self, X, Y, mini_batch_size):
        m = X.shape[1]
        mini_batches = []
        
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            k = m // mini_batch_size
            mini_batch_X = shuffled_X[:, k*mini_batch_size :]
            mini_batch_Y = shuffled_Y[:, k*mini_batch_size :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches