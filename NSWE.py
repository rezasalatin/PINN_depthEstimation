"""
PINN for reconstructing UV field with nonlinear shallow water equations and lab data
@author: Reza Salatin
November 2023
"""

import sys                                                  # This import statement is used to access system-specific parameters and functions in Python.
import tensorflow as tf                                     # building/training neural networks in machine learning and deep learning.
import numpy as np                                          # numerical operations with large, multi-dimensional arrays and matrices.
#import matplotlib.pyplot as plt                             # plotting graphs and visualizing data.
import scipy.io                                             # reading and writing MATLAB files.
#from scipy.interpolate import griddata                      # interpolating data points on a grid.
#from pyDOE import lhs                                       # generating Latin Hypercube Samples for design of experiments.
#from plotting import newfig, savefig                        # creating and saving figures in a specific format.
#from mpl_toolkits.mplot3d import Axes3D                     # 3D plotting capabilities in Matplotlib.
import time                                                 # accessing time-related functions, like delays or time measurement.
#import matplotlib.gridspec as gridspec                      # creating grid layouts for subplots in Matplotlib.
#from mpl_toolkits.axes_grid1 import make_axes_locatable     # dividing axes in Matplotlib plots to place colorbars.

sys.path.insert(0, '../utilities/')                         # Adds '../utilities/' to the beginning of the system path list for module import resolution.
np.random.seed(1234)                                        # Sets a fixed seed for NumPy's random number generation, ensuring reproducibility.
tf.set_random_seed(1234)                                    # Sets a fixed seed for TensorFlow's random number generation, ensuring consistent results in machine learning models.

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
################################ PINN ################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# 

class PhysicsInformedNN:
    # This class initializes a physics-informed neural network.

    def __init__(self, t, x, y, h, u, v, z, layers):
        # The constructor combines inputs into a single array, initializes network layers, weights, biases, and placeholders.

        # inputs
        self.t = t
        self.x, self.y, self.h = x, y, h
        # outputs
        self.u, self.v, self.z = u, v, z
        # parameters
        self.layers = layers

        # Initialize neural network weights and biases.
        self.weights, self.biases = self.initialize_NN(layers)

        # TensorFlow placeholders for feeding data into the network.
        self.t_tf = [tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])]
        self.x_tf, self.y_tf, self.h_tf = [tf.placeholder(tf.float32, shape=[None, item.shape[1]]) for item in [self.x, self.y, self.h]]
        self.u_tf, self.v_tf, self.z_tf = [tf.placeholder(tf.float32, shape=[None, item.shape[1]]) for item in [self.u, self.v, self.z]]

        # Define the neural network function and predicted outputs.
        self.u_pred, self.v_pred, self.z_pred, self.f_u_pred, self.f_v_pred, self.f_c_pred = self.net_NSWE(self.t_tf, self.x_tf, self.y_tf, self.h_tf)

        # Loss function combining mean squared errors of (actuals - predictions) and residuals.
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_mean(tf.square(self.z_tf - self.z_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred)) + \
                    tf.reduce_mean(tf.square(self.f_c_pred))
        
        # Optimization: L-BFGS-B and Adam optimizer.
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', options = {'maxiter': 50000, 'maxfun': 50000, 'maxcor': 50, 'maxls': 50, 'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # TensorFlow session for executing the graph.
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        
        # Initialize global variables.
        init = tf.global_variables_initializer()
        self.sess.run(init)
             

    ######################## NN Initialization ##########################

    def initialize_NN(self, layers):
        # This function initializes weights and biases for each layer in the neural network.

        weights = []  # List to store weights of each layer.
        biases = []   # List to store biases of each layer.
        num_layers = len(layers)  # Total number of layers in the network.

        for l in range(0, num_layers - 1):
            # Iterates through each layer to initialize weights and biases.

            W = self.xavier_init(size=[layers[l], layers[l + 1]])  # Initialize weights using Xavier initialization.
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)  # Initialize biases to zeros.

            weights.append(W)  # Add weights to the list.
            biases.append(b)   # Add biases to the list.

        return weights, biases  # Return lists of weights and biases.     


    ######################## Xavier Initialization ##########################
    
    def xavier_init(self, size):
        # This function implements the Xavier initialization for weights in a neural network layer.

        in_dim = size[0]  # The number of input units in the weight matrix.
        out_dim = size[1] # The number of output units in the weight matrix.

        # Calculate the standard deviation for Xavier/Glorot initialization.
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))

        # Return TensorFlow variable initialized with a truncated normal distribution based on Xavier standard deviation.
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    
    ############################## NN ##########################

    def neural_net(self, X, weights, biases):
        # This function constructs the neural network using the provided weights and biases.

        num_layers = len(weights) + 1  # Calculate the total number of layers in the network.

        # H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 # Normalization of input data (commented out)
        H = X  # Directly using the input data without normalization.

        for l in range(0, num_layers - 2):
            # Loop through each layer except the last one.
            W = weights[l]  # Weights for the current layer.
            b = biases[l]   # Biases for the current layer.

            # Apply a linear transformation followed by a tanh activation function.
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        W = weights[-1]  # Weights for the last layer.
        b = biases[-1]   # Biases for the last layer.

        # Output of the neural network without an activation function for the last layer.
        Y = tf.add(tf.matmul(H, W), b)

        return Y  # Return the output of the neural network.


    ############################## Physics ##########################

    def net_NSWE(self, t, x, y, h):
        # This function defines the physics-informed neural network for the NSWE (Navier-Stokes Wave Equation).

        # Pass inputs through neural network to get predictions for u, v, and e.
        uvz = self.neural_net(tf.concat([t, x, y, h], 1), self.weights, self.biases)
        u, v, z = uvz[:, 0:1], uvz[:, 1:2], uvz[:, 2:3]

        # Calculate temporal and spatial gradients for u, v, and z.
        u_t, u_x, u_y = [tf.gradients(u, var)[0] for var in [t, x, y]]
        v_t, v_x, v_y = [tf.gradients(v, var)[0] for var in [t, x, y]]
        z_t, z_x, z_y = [tf.gradients(z, var)[0] for var in [t, x, y]]

        # Define the physics-informed functions f_u, f_v, and f_c based on the gradients and equations.
        f_u = u_t + (u * u_x + v * u_y) + 9.81 * (z_x + z_y)
        f_v = v_t + (u * v_x + v * v_y) + 9.81 * (z_x + z_y)
        f_c = z_t + (u * z_x + v * z_y) + (h + z) * (u_x + v_y)

        return u, v, z, f_u, f_v, f_c

    def callback(self, loss):
        # Callback function to print the loss during training.
        print('Loss:', loss)


    ############################## Training ##########################

    def train(self, nIter):
        # Function to train the neural network.

        # Define TensorFlow dictionary with input and output placeholders.
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x, self.y_tf: self.y, self.u_tf: self.u, self.v_tf: self.v, self.z_tf: self.z}

        # Training loop.
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print loss every 10 iterations.
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % (it, loss_value, elapsed))
                start_time = time.time()

        # Final optimization using L-BFGS-B optimizer.
        self.optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=[self.loss], loss_callback=self.callback)


    ############################## Predict ##########################

    def predict(self, t_star, x_star, y_star, h_star):
        # Function to make predictions using the trained model.

        # Define TensorFlow dictionary for prediction.
        tf_dict = {self.t_tf: t_star, self.x_tf: x_star, self.y_tf: y_star, self.h_tf: h_star}

        # Run the session to get predictions.
        u_star, v_star, z_star = self.sess.run([self.u_pred, self.v_pred, self.z_pred], tf_dict)
        f_u_star, f_v_star, f_c_star = self.sess.run([self.f_u_pred, self.f_v_pred, self.f_c_pred], tf_dict)

        return u_star, v_star, z_star, f_u_star, f_v_star, f_c_star


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
################################ Main ################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

if __name__ == "__main__": 
    # The main execution block of the script.

    # Number of training points on domain.
    N_train = 1000

    # Neural network configuration: inputs, hidden layers with multiple nodes each, and outputs.
    # info    i  1   2   3   4   5   6   7   8   o  (input, hidden layer #, and output)
    layers = [4, 20, 20, 20, 20, 20, 20, 20, 20, 3] # number of parameters/weights

    # Load data from a .mat file.
    data = scipy.io.loadmat('../data/exp.mat')

    # Extracting and rearranging data for input into the neural network.
    t_star = data['t']                                          # Time,
    X_star, Y_star, h_star = data['X'], data['Y'], data['h']    # Spatial coordinates and water depth.
    U_star, V_star, Z_star = data['U'], data['V'], data['Z']    # Velocity and eta.
    N, T = X_star.shape[0], t_star.shape[0]

    # Preprocessing and flattening data for neural network training.
    TT = np.tile(t_star, (1,N)).T
    XX, YY, hh = np.tile(X_star, (1,T)), np.tile(Y_star, (1,T)), np.tile(h_star, (1,T))
    UU, VV, ZZ = np.tile(U_star, (1,T)), np.tile(V_star, (1,T)), np.tile(Z_star, (1,T))

    t = TT.flatten()[:,None]
    x, y, h = XX.flatten()[:,None], YY.flatten()[:,None], hh.flatten()[:,None]
    u, v, z = UU.flatten()[:,None], VV.flatten()[:,None], ZZ.flatten()[:,None]

    # Selecting a subset of data for training.
    idx = np.random.choice(N*T, N_train, replace=False)
    t_train = t[idx,:]
    x_train, y_train, h_train = x[idx,:], y[idx,:], h[idx,:]
    u_train, v_train, z_train = u[idx,:], v[idx,:], z[idx,:]

    # Initializing and training the neural network model.
    model = PhysicsInformedNN(t_train, x_train, y_train, h_train, u_train, v_train, z_train, layers)
    start_time = time.time()   
    model.train(50000)
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)

    # Save the model
    saver = tf.train.Saver()
    save_path = saver.save(model.sess, "./model.ckpt")
    print("Model saved in path: %s" % save_path)

    ############################## Testing Data ##########################
    # Setting up testing data for model evaluation.
    snap = np.array([100])
    t_star = TT[:,snap]
    x_star, y_star, h_star = X_star, Y_star, h_star
    u_star, v_star, z_star = U_star[:,snap], V_star[:,snap], Z_star[:,snap]

    # Making predictions using the trained model.
    u_pred, v_pred, z_pred, f_u_pred, f_v_pred, f_c_pred = model.predict(t_star, x_star, y_star, h_star)

    # Save the testing locations and prediction results
    np.save('./x_test.npy', x_star)
    np.save('./y_test.npy', y_star)
    np.save('./u_pred.npy', u_pred)
    np.save('./v_pred.npy', v_pred)
    np.save('./z_pred.npy', z_pred)


    # Calculating errors between predictions and actual data.
    error_u, error_v, error_z = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2), np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2), np.linalg.norm(z_star-z_pred,2)/np.linalg.norm(z_star,2)
    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error z: %e' % error_z)