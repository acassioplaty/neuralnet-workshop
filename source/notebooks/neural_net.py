import numpy as np

class Activation():
    def __init__(self):
        pass
    
    def f_activ_sig(self, x):
        return 1 / (1 + np.exp(-x))

    def f_activ_sig_prime(self, x):
        return self.f_activ_sig(x)*(1-self.f_activ_sig(x))
    
    def f_activ_relu(self, x):
        return (abs(x) + x)/2

    def f_activ_relu_prime(self, x):
        return np.where(x > 0, 1.0, 0.0)

    def f_ident(self, x):
        return x
    
    def f_ident_prime(self, x):
        return np.ones_like(x)
    
    def f_tanh(self, x):
        return np.tanh(x)
    
    def f_tanh_prime(self, x):
        return 1 - np.tanh(x)**2
    
    def softmax(self, x):
        if x.ndim != 1:
            return np.exp(x) / np.sum(np.exp(x), axis = 1)[:, None]
        else:
            return np.exp(x) / np.sum(np.exp(x), axis = 0)
    
    def softmax_prime(self, x):
        s = self.softmax(x).reshape(-1,1)
        return np.diag(np.diagflat(s) - np.dot(s, s.T))
    
class Layer():
    def __init__(self, neurons, type_, activation="relu"):
        activ = Activation()
        self.neurons = neurons
        self.type_ = type_
        self.activation_name = activation
        if type_ == "input":
            self.activation = activ.f_ident
            self.activation_prime = activ.f_ident_prime
            self.activation_name = "ident"
        elif activation == "relu":
            self.activation = activ.f_activ_relu
            self.activation_prime = activ.f_activ_relu_prime
        elif activation == "tanh":
            self.activation = activ.f_tanh
            self.activation_prime = activ.f_tanh_prime
        elif activation == "ident":
            self.activation = activ.f_ident
            self.activation_prime = activ.f_ident_prime   
        elif activation == "softmax":
            self.activation = activ.softmax
            self.activation_prime = activ.softmax_prime               
        else:
            self.activation = activ.f_activ_sig
            self.activation_prime = activ.f_activ_sig_prime


class MLP():
    def __init__(self, layers):
        self.layers = {k: item for k, item in enumerate(layers)}
        
        self.weights = dict()
        self.biases = dict()       
        
        init_ = 0
        end_ = len(self.layers)
        
        for i, k in zip(range(0,end_-1), range(init_+1, end_)):
            dim_1 = self.layers[i].neurons
            dim_2 = self.layers[k].neurons
            self.weights[k] = np.random.normal(0, dim_1 ** -.5,
                                        size=(dim_1, dim_2))
            
            self.biases[k] = np.zeros(dim_2)
    
    def summary(self):
        for k, layer in self.layers.items():
            print("Layer {} - {} com {} neurônios e funcão de ativação {}".format(layer.type_,
                                                                                  k, layer.neurons, layer.activation_name))
            
        for k, weight in self.weights.items():
            print("Shape matriz {} = {} / Shape bias {} = {}". format(k, weight.shape, k, self.biases[k].shape))
            
    def forward(self, x):
        input_ = x
        
        hidden = dict()
        primes = dict()
        
        init_ = 0
        end_ = len(self.layers)
        
        for i, k in zip(range(0,end_-1), range(init_+1, end_)):
            dot_product = np.dot(input_, self.weights[k]) + self.biases[k]
            hidden[k] = self.layers[k].activation(dot_product)
            primes[k] = self.layers[k].activation_prime(dot_product)
            input_ = hidden[k]    
            
        return hidden, primes
    
    def init_steps(self):
        steps = dict()
        biases_steps=dict()
        for k in range(1, len(self.layers), 1):
            steps[k] = np.zeros((self.weights[k].shape[0], self.weights[k].shape[1]))
            biases_steps[k] = np.zeros(self.weights[k].shape[1])    
        
        return steps, biases_steps
    
    def train(self, X, Y, batch, epochs=1000, learnrate=0.5):
        last_loss = None
        for e in range(epochs):
            steps, biases_steps = self.init_steps()
            for x, y in zip(X, Y):
                hidden, primes = self.forward(x)
                
                error_term = dict()
                                
                init_ = len(self.layers) - 1 
                end_ = 0
                
                for k in range(init_, 0, -1):
                    if k == init_:
                        error = -(y-hidden[k])
                    else:
                        error = np.dot(error_term[k+1], self.weights[k+1].T)
                        
                    error_term[k] =error*primes[k]
                    
                    if k != end_ + 1:
                        steps[k] += error_term[k]*hidden[k-1][:, None]

                    else:
                        steps[k] += error_term[k]*x[:, None]
                        

                    biases_steps[k] += error_term[k]
                    
            for k in range(1, len(self.layers), 1):
                self.weights[k] -= learnrate*steps[k] / len(X)    
                self.biases[k] -= learnrate*biases_steps[k] / len(X)
            
            if e % (epochs / 10) == 0:
                input_ = X
                for k in range(1, len(self.layers), 1):
                    output_ = self.layers[k].activation(np.dot(input_, self.weights[k]) + self.biases[k])
                    input_ = output_

                loss_f = np.mean((output_ - Y) ** 2)
                
                
                if last_loss and last_loss < loss_f:
                    print("Train loss: ", loss_f, "  WARNING - Loss Increasing")
                else:
                    print("Train loss: ", loss_f)
                last_loss = loss_f

    def predict(self, X):
        input_ = X
        for k in range(1, len(self.layers), 1):
            output_ = self.layers[k].activation(np.dot(input_, self.weights[k]) + self.biases[k])
            input_ = output_
            
        return output_