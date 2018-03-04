# pkML
Small machine learning library of self-written algorithms.

## NeuralNet

Neural net class for quick prototyping

### Parameters


<code><b>layer_dimensions</b>: 1D array-like (int), shape = [n_layers]</code><br>
        Used for specifying the number of neurons per layer. Last element 
        needs to be 1 due to the fact that multiclass-classification isn't 
        supported so far.<br></t>
<code><b>activations</b>: 1D array-like (string), shape = [n_layers]</code> <br>
        Activation functions for each layer. Supported activation functions
        are 'sigmoid', 'tanh', 'relu' and 'leaky relu'. Last element should most likely be
        'sigmoid'.<br>
<code><b>optimizer</b>: string, optional (default='SGD')</code> <br>
        Optimizer algorithm for parameter updating. Default is Stochastic 
        Gradient Descent ('SGD'). Other supported option is Adaptive 
        Momentum Estimation ('Adam').<br>

### Methods

#### <code>fit(X, y, epochs = 10, learning_rate = 0.1, lambda_l1 = 0, lambda_l2 = 1.0, batch_size = None,random_seed = None, warm_start = False, verbose = 1)</code>

Train the neural net from the training set (X, y).<br>

<code><b>X</b>: array-like, shape = [n_samples, n_features]</code> <br>
            The training input samples. <br>
<code><b>y</b>: 1D array-like, shape = [n_samples]</code> <br>
            The target values.<br>
<code><b>epochs</b>: int, optional (default=10)</code> <br>
            Number of training iterations on the full dataset. <br>
<code><b>learning_rate</b>: float, optional (default=0.1)</code> <br>
            Parameter that defines the size of the optimization algorithm steps.<br>
<code><b>lambda_l1</b>: float, optional (default=0.0)</code> <br>
            L1 regularization parameter. Larger values increase the 
            regularization effect. <br>
<code><b>lambda_l2</b>: float, optional (default=1.0)</code> <br>
            L2 regularization parameter. Larger values increase the 
            regularization effect. <br>
<code><b>batch_size</b>: int, optional (default=None)</code> <br>
            Size of the batch that is computed before the weights are updated.
            If no value is given, batch optimization is used. Smaller values
            result in faster convergence of the loss function per epoch. Too 
            small values may slow down the training. Typical values are 64,
            128, 256, 512, 1024.<br>
<code><b>random_seed</b>: int, optinal (default=None)</code> <br>
            Seed used by the random number generater for the parameter 
            initialization. <br>
<code><b>warm_start</b>: bool, optional (default=False)</code> <br>
            When set to 'True', reuse the solution of the previous call to fit
            and update the already training parameters.<br>
<code><b>verbose</b>: int, optional (default=1)</code> <br>
            Controls the verbosity of the training process. <br>

#### <code>predict_proba(X)</code>

Predict class probabilities for X.<br>

<code><b>X</b>: array-like, shape = [n_samples, n_features]</code> <br>
            The input samples. <br>
            
##### returns
<code><b>p</b>: array-like, shape = [n_samples]</code> <br>
            The class probabilities of the input samples. <br>
