import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def leaky_relu(x):
    return np.maximum(0.1*x, x)

class NeuralNet:
    """ Simple artificial neural net class for quick prototyping. 
    
    Only supports fully-connected layers. Uses Xavier initialization
    
    Parameters
    ----------
    layer_dimensions : 1D array-like (int), shape = [n_layers]
        Used for specifying the number of neurons per layer. Last element 
        needs to be 1 due to the fact that multiclass-classification isn't 
        supported so far.
    activations : 1D array-like (string), shape = [n_layers]
        Activation functions for each layer. Supported activation functions
        are 'sigmoid', 'tanh', 'relu' and 'leaky relu'. Last element should most likely be
        'sigmoid'.
    optimizer : string, optional (default='SGD')
        Optimizer algorithm for parameter updating. Default is Stochastic 
        Gradient Descent ('SGD'). Other supported option is Adaptive 
        Momentum Estimation ('Adam'). 
    """
    def __init__(self, layer_dimensions, activations, optimizer = 'SGD'):
        assert( len(layer_dimensions) == len(activations) )
        assert( (len(layer_dimensions) > 0) & (len(activations) > 0) )
        assert(np.mean([(a in ['sigmoid', 'tanh', 'relu', 'leaky relu']) for a in activations]) == 1)
        assert(optimizer in ['SGD', 'Adam'])
        
        self.layer_dims = layer_dimensions
        self.activations = activations
        self.gradients = {}
        self.parameters_W = {}
        self.parameters_b = {}
        self.cache = {}
        self.optimizer = optimizer
        self.costs = []
        
        if optimizer is 'Adam':
            self.v = {}
            self.s = {}
            self.v_corr = {}                         
            self.s_corr = {}
            self.beta1 = 0.9
            self.beta2 = 0.999  
            self.epsilon = 1e-8
    
    def logloss(self, p, y, lambda_l1 = 1, lambda_l2 = 1):
        assert(len(p) == len(y))
        
        # Cross-Entropy cost
        cross_entropy_cost = - np.sum( np.multiply(y, np.log(p)) + np.multiply((1-y), np.log(1-p)) ) / y.shape[1]
        
        # L1 regularization cost
        if lambda_l1 > 0:
            reg_cost_l1 = np.sum([(np.sum(np.abs(arr))) for arr in self.parameters_W.values()]) / (y.shape[1])
        else:
            reg_cost_l1 = 0
            
        # L2 regularization cost
        if lambda_l2 > 0:
            reg_cost_l2 = np.sum([(np.sum(np.square(arr))) for arr in self.parameters_W.values()]) / (2*y.shape[1])
        else: 
            reg_cost_l2 = 0
        
        return cross_entropy_cost + np.multiply(lambda_l1, reg_cost_l1) + np.multiply(lambda_l2, reg_cost_l2)

    def fit(self, X, y, epochs = 10, learning_rate = 0.1, lambda_l1 = 0, lambda_l2 = 1.0, batch_size = None,
            random_seed = None, warm_start = False, verbose = 1):
        """ Train the neural from the training set (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. 
        y : 1D array-like, shape = [n_samples]
            The target values.
        epochs : int, optional (default=10)
            Number of training iterations on the full dataset. 
        learning_rate : float, optional (default=0.1)
            Parameter that defines the size of the optimization algorithm steps.
        lambda_l1 : float, optional (default=0.0)
            L1 regularization parameter. Larger values increase the 
            regularization effect. 
        lambda_l2 : float, optional (default=1.0)
            L2 regularization parameter. Larger values increase the 
            regularization effect.
        batch_size : int, optional (default=None)
            Size of the batch that is computed before the weights are updated.
            If no value is given, batch optimization is used. Smaller values
            result in faster convergence of the loss function per epoch. Too 
            small values may slow down the training. Typical values are 64,
            128, 256, 512, 1024.
        random_seed : int, optinal (default=None)
            Seed used by the random number generater for the parameter 
            initialization.
        warm_start : bool, optional (default=False)
            When set to 'True', reuse the solution of the previous call to fit
            and update the already training parameters.
        verbose : int, optional (default=1)
            Controls the verbosity of the training process.
        """
        X = X.T
        
        # If batch_size isn't specified, use full dataset as batch
        if batch_size is None:
            batch_size = X.shape[1]
        
        if not warm_start:
            np.random.seed(random_seed)
            
            for l in range(len(self.layer_dims)):
                
                # Initialize weight parameters 
                if l == 0:
                    self.parameters_W[l] = np.random.randn(self.layer_dims[l], X.shape[0]) * np.sqrt(1.0 / X.shape[0])
                    self.parameters_b[l] = np.zeros((self.layer_dims[l], 1))
                else:
                    self.parameters_W[l] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(1.0 / layer_dims[l-1])
                    self.parameters_b[l] = np.zeros((self.layer_dims[l], 1))
                
                if self.optimizer is 'Adam':
                    self.t = 0 # Adam counter
                    self.v["dW" + str(l)] = np.zeros(self.parameters_W[l].shape)
                    self.v["db" + str(l)] = np.zeros(self.parameters_b[l].shape)
                    self.s["dW" + str(l)] = np.zeros(self.parameters_W[l].shape)
                    self.s["db" + str(l)] = np.zeros(self.parameters_b[l].shape)    

        for epoch in range(epochs):
            for batch in range(int(X.shape[1] / batch_size + 1)):
                if batch_size*batch < X.shape[1]:
                    X_batch = X[:, batch_size*batch:batch_size*(batch+1)]
                    y_batch = y[:, batch_size*batch:batch_size*(batch+1)]
                    self.cache['a-1'] = X_batch

                    # Forward propagation
                    p = self.forward_prop(X_batch)

                    # Calculate cost
                    cost = self.logloss(p, y_batch, lambda_l1, lambda_l2)
                    self.costs.append(cost)

                    # Backward propagation
                    self.backward_prop(p, y_batch, lambda_l1, lambda_l2)
                    
                    # Update parameters
                    self.parameter_update(learning_rate)

            if verbose >= 1:    
                cost = self.logloss(self.forward_prop(X), y, lambda_l1, lambda_l2)
                print('Cost' + str(epoch) + ' : ' + str(cost))
       
    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples. 

        Returns
        -------
        p : array-like, shape = [n_samples]
            The class probabilities of the input samples. 
        """
        return self.forward_prop(X.T).T
        
    def forward_prop_single_step(self, X, W, b):
        return np.dot(W, X) + b

    def forward_prop(self, X):
        
        a = X
        for l in range(len(self.layer_dims)):
            W = self.parameters_W[l]
            b = self.parameters_b[l]
            z = self.forward_prop_single_step(a, W, b)
            
            if self.activations[l] == 'sigmoid':
                a = sigmoid(z)
            elif self.activations[l] == 'tanh':
                a = np.tanh(z)
            elif self.activations[l] == 'relu':
                a = relu(z)
            elif self.activations[l] == 'leaky relu':
                a = leaky_relu(z)
            
            self.cache['z' + str(l)] = z
            self.cache['a' + str(l)] = a
    
        return a
    
    def backward_prop(self, A_l, y, lambda_l1, lambda_l2):
        y = y.reshape(A_l.shape)
        L = len(self.layer_dims) # number of layers
        
        dA_l = - (np.divide(y, A_l) - np.divide(1 - y, 1 - A_l))
        
        l = L-1
        dA, dW, db = self.backward_prop_single_step(dA_l, self.cache['z' + str(l)], l, activations[L-1], 
                                                    lambda_l1, lambda_l2)
        self.gradients['dA' + str(l)] = dA
        self.gradients['dW' + str(l)] = dW
        self.gradients['db' + str(l)] = db
        
        for l in reversed(range(L-1)):
            dA, dW, db = self.backward_prop_single_step(dA, self.cache['z' + str(l)], l, activations[l], 
                                                        lambda_l1, lambda_l2)
            self.gradients['dA' + str(l)] = dA
            self.gradients['dW' + str(l)] = dW
            self.gradients['db' + str(l)] = db
            
    
    def backward_prop_single_step(self, dA, z, l, activation, lambda_l1, lambda_l2):
        A_prev = self.cache['a' + str(l-1)]
        W = self.parameters_W[l]
        
        if activation == 'sigmoid':
            temp = 1 / (1 + np.exp(-z))
            dz = dA * temp * (1-temp)
        elif activation == 'tanh':
            dz = dA * 2 / (np.cosh(2*z) + 1)
        elif activation == 'relu':
            dz = np.array(dA, copy=True)
            dz[z <= 0] = 0
        elif activation == 'leaky relu':
            dz = np.array(dA, copy=True)
            dz[z < 0] = 0.1 * dz[z < 0]
        
        m = A_prev.shape[1]
        dW = 1.0/m * (np.dot(dz, A_prev.T) + lambda_l1*np.sign(W)  + lambda_l2 * W)
        db = 1.0/m * np.sum(dz, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dz)
            
        assert(dz.shape == z.shape)
        
        return dA_prev, dW, db
        
    def parameter_update(self, learning_rate):
        
        for l in range(len(layer_dims)):
            if self.optimizer is 'SGD':
                self.parameters_W[l] = self.parameters_W[l] - learning_rate * self.gradients['dW' + str(l)]
                self.parameters_b[l] = self.parameters_b[l] - learning_rate * self.gradients['db' + str(l)]
                        
            elif self.optimizer is 'Adam':
                self.t = self.t + 1 # Adam counter
                            
                self.v["dW" + str(l)] = self.beta1 * self.v["dW" + str(l)] + (1-self.beta1) * self.gradients['dW'+ str(l)]
                self.v["db" + str(l)] = self.beta1 * self.v["db" + str(l)] + (1-self.beta1) * self.gradients['db'+ str(l)]

                self.v_corr["dW" + str(l)] = self.v["dW" + str(l)] / (1 - self.beta1**self.t)
                self.v_corr["db" + str(l)] = self.v["db" + str(l)] / (1 - self.beta1**self.t)
                            
                self.s["dW" + str(l)] = self.beta2 * self.s["dW" + str(l)] + (1-self.beta2) * np.square(self.gradients['dW'+ str(l)])
                self.s["db" + str(l)] = self.beta2 * self.s["db" + str(l)] + (1-self.beta2) * np.square(self.gradients['db'+ str(l)])
                            
                self.s_corr["dW" + str(l)] = self.s["dW" + str(l)] / (1 - self.beta2**self.t)
                self.s_corr["db" + str(l)] = self.s["db" + str(l)] / (1 - self.beta2**self.t)
                            
                self.parameters_W[l] = self.parameters_W[l] - learning_rate * self.v_corr["dW" + str(l)] / (np.sqrt(self.s_corr["dW" + str(l)]) + self.epsilon)
                self.parameters_b[l] = self.parameters_b[l] - learning_rate * self.v_corr["db" + str(l)] / (np.sqrt(self.s_corr["db" + str(l)]) + self.epsilon)
 
