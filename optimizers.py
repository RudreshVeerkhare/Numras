import numpy as np
from sklearn.metrics import accuracy_score
import math

class AdamOptimizer:
    def __init__(self, B1=0.9, B2=0.999, epsilon=1e-7, learning_rate=0.001):
        self.B1 = B1
        self.B2 = B2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.Vd = dict()
        self.Sd = dict()
        self.L = None
        self.t = 0
        
    def init_params(self, layers):
        self.L = len(layers)
        for l in range(1, self.L):
            self.Vd[f"W{l}"] = np.zeros((layers[l], layers[l-1]))
            self.Vd[f"b{l}"] = np.zeros((layers[l], 1))
            self.Sd[f"W{l}"] = np.zeros((layers[l], layers[l-1]))
            self.Sd[f"b{l}"] = np.zeros((layers[l], 1))
    
    def take_step(self, derivatives):
        final_derivatives = dict()
        for l in range(1, self.L):
            self.t += 1
            self.Vd[f"W{l}"] = self.B1 * self.Vd[f"W{l}"] + (1 - self.B1) * derivatives[f"dW{l}"]
            self.Vd[f"b{l}"] = self.B1 * self.Vd[f"b{l}"] + (1 - self.B1) * derivatives[f"db{l}"]
            self.Sd[f"W{l}"] = self.B2 * self.Sd[f"W{l}"] + (1 - self.B2) * derivatives[f"dW{l}"]**2
            self.Sd[f"b{l}"] = self.B2 * self.Sd[f"b{l}"] + (1 - self.B2) * derivatives[f"db{l}"]**2
            
            lr = self.learning_rate * np.sqrt(1 - np.power(self.B2, self.t)) / (1 - np.power(self.B1, self.t))
            
            final_derivatives[f"dW{l}"] = lr * self.Vd[f"W{l}"] / (np.sqrt(self.Sd[f"W{l}"]) + self.epsilon)
            final_derivatives[f"db{l}"] = lr * self.Vd[f"b{l}"] / (np.sqrt(self.Sd[f"b{l}"]) + self.epsilon)
            
#         print(list(final_derivatives.items())[0])
            
        return final_derivatives
    
class Momentum:
    def __init__(self, B1=0.9, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.B1 = B1
        self.Vd = dict()
        self.L = None
        self.t = 0
    
    def init_params(self, layers):
        self.L = len(layers)
        for l in range(1, self.L):
            self.Vd[f"W{l}"] = np.zeros((layers[l], layers[l-1]))
            self.Vd[f"b{l}"] = np.zeros((layers[l], 1))

    def take_step(self, derivatives):
        final_derivatives = dict()
        for l in range(1, self.L):
            self.t += 1
            self.Vd[f"W{l}"] = (self.B1 * self.Vd[f"W{l}"] + self.learning_rate * derivatives[f"dW{l}"]) #/ (1 - np.power(self.B1, self.t))
            self.Vd[f"b{l}"] = (self.B1 * self.Vd[f"b{l}"] + self.learning_rate * derivatives[f"db{l}"]) #/ (1 - np.power(self.B1, self.t))
            
            
            final_derivatives[f"dW{l}"] = self.Vd[f"W{l}"]
            final_derivatives[f"db{l}"] = self.Vd[f"b{l}"]
            
#         print(list(final_derivatives.items())[0])
            
        return final_derivatives
    
    
class RMSProp:
    def __init__(self, B2=0.9, epsilon=1e-7, learning_rate=0.001):
        self.B2 = B2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.Sd = dict()
        self.L = None
        self.t = 0
       
    def init_params(self, layers):
        self.L = len(layers)
        for l in range(1, self.L):
            self.Sd[f"W{l}"] = np.zeros((layers[l], layers[l-1]))
            self.Sd[f"b{l}"] = np.zeros((layers[l], 1))
   
    def take_step(self, derivatives):
        final_derivatives = dict()
        for l in range(1, self.L):
            self.t += 1
            self.Sd[f"W{l}"] = (self.B2 * self.Sd[f"W{l}"] + (1 - self.B2) * derivatives[f"dW{l}"]**2)  / (1 - np.power(self.B2, self.t))
            self.Sd[f"b{l}"] = (self.B2 * self.Sd[f"b{l}"] + (1 - self.B2) * derivatives[f"db{l}"]**2)  / (1 - np.power(self.B2, self.t))
            
            
            
            final_derivatives[f"dW{l}"] = self.learning_rate * derivatives[f"dW{l}"] / (np.sqrt(self.Sd[f"W{l}"]) + self.epsilon)
            final_derivatives[f"db{l}"] = self.learning_rate * derivatives[f"db{l}"] / (np.sqrt(self.Sd[f"b{l}"]) + self.epsilon)
            
#         print(list(final_derivatives.items())[0])
            
        return final_derivatives
    

class AdaDelta:
    def __init__(self, B2=0.9, epsilon=1e-7):
        self.B2 = B2
        self.epsilon = epsilon
        self.Vd = dict()
        self.Wd = dict()
        self.L = None
        self.t = 0
        
    def init_params(self, layers):
        self.L = len(layers)
        for l in range(1, self.L):
            self.Vd[f"W{l}"] = np.zeros((layers[l], layers[l-1]))
            self.Vd[f"b{l}"] = np.zeros((layers[l], 1))
            self.Wd[f"W{l}"] = np.zeros((layers[l], layers[l-1]))
            self.Wd[f"b{l}"] = np.zeros((layers[l], 1))
    
    def take_step(self, derivatives):
        final_derivatives = dict()
        for l in range(1, self.L):
            self.t += 1
            self.Vd[f"W{l}"] = self.B2 * self.Vd[f"W{l}"] + (1 - self.B2) * np.square(derivatives[f"dW{l}"])
            self.Vd[f"b{l}"] = self.B2 * self.Vd[f"b{l}"] + (1 - self.B2) * np.square(derivatives[f"db{l}"])
            
            dW = (np.sqrt(self.Wd[f"W{l}"] + self.epsilon)) / (np.sqrt(self.Vd[f"W{l}"] + self.epsilon))
            db = (np.sqrt(self.Wd[f"b{l}"] + self.epsilon)) / (np.sqrt(self.Vd[f"b{l}"] + self.epsilon))
            
            self.Wd[f"W{l}"] = self.B2 * self.Wd[f"W{l}"] + (1 - self.B2) * np.square(dW)
            self.Wd[f"b{l}"] = self.B2 * self.Wd[f"b{l}"] + (1 - self.B2) * np.square(db)
            
            final_derivatives[f"dW{l}"] = dW
            final_derivatives[f"db{l}"] = db
            
#         print(list(final_derivatives.items())[0])
            
        return final_derivatives