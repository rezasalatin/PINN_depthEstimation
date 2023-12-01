"""
PINN for NSWE
@author: Reza Salatin
December 2023
w/ Tensorflow 2
"""

import tensorflow as tf
import numpy as np
import time
from l_bfgs_b_optimizer import LBFGSBOptimizer

np.random.seed(1234)
tf.random.set_seed(1234)

###############################################################################
# General access functions ####################################################

def custom_loss(model, inputs, true_vals):

    u_pred, v_pred, z_pred, f_u_pred, f_v_pred, f_c_pred = model.net_NSWE(inputs)
    u_true, v_true, z_true = tf.split(true_vals, num_or_size_splits=3, axis=-1)
    loss = tf.reduce_mean(tf.square(u_true - u_pred)) + \
           tf.reduce_mean(tf.square(v_true - v_pred)) + \
           tf.reduce_mean(tf.square(z_true - z_pred)) + \
           tf.reduce_mean(tf.square(f_u_pred)) + \
           tf.reduce_mean(tf.square(f_v_pred)) + \
           tf.reduce_mean(tf.square(f_c_pred))
    return loss

def train_step(model, inputs, true_vals, optimizer):
    with tf.GradientTape() as tape:
        loss = custom_loss(model, inputs, true_vals)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

###############################################################################
# PINN Class ##################################################################

class PhysicsInformedNN(tf.keras.Model):
    def __init__(self, layers):
        super(PhysicsInformedNN, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(layer, activation=tf.nn.tanh) 
                             for layer in layers[:-1]]
        self.output_layer = tf.keras.layers.Dense(layers[-1], activation=None)
        
    def call(self, inputs):
        # Forward pass through the network
        x = inputs
        for dense in self.dense_layers:
            x = dense(x)
        return self.output_layer(x)

# L-BFGS-B ####################################################################
    def train_with_lbfgs(self, inputs, outputs):
        lbfgs_optimizer = LBFGSBOptimizer(self, inputs, outputs, custom_loss, options={'maxiter': 50, 'maxfun': 50000, 'maxcor': 50, 'maxls': 50, 'ftol' : 1.0 * np.finfo(float).eps})
        lbfgs_optimizer.optimize()

# Training ####################################################################
    # In the train method of the PhysicsInformedNN class
    def train(self, inputs, true_vals, nIter):
        optimizer = tf.keras.optimizers.Adam()
        for it in range(nIter):
            loss = train_step(self, inputs, true_vals, optimizer)
            if it % 10 == 0:
                print(f'Iteration {it}, Loss: {loss.numpy()}')
        # No need for a separate L-BFGS-B optimizer step in this simplified example

# Prediction ##################################################################
    def predict(self, inputs):
        # Make predictions using the model
        u, v, z, f_u, f_v, f_c = self.net_NSWE(inputs)
        return u, v, z, f_u, f_v, f_c

# Physics ##################################################################
    def net_NSWE(self, inputs):

        t, x, y, h = tf.split(inputs, num_or_size_splits=4, axis=-1)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([t, x, y, h])
            uvz = self.call(tf.concat([t, x, y, h], axis=1))
            u, v, z = uvz[:, 0:1], uvz[:, 1:2], uvz[:, 2:3]

        # Calculate temporal and spatial gradients for u, v, and z.
        u_t = tape.gradient(u, t)
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)

        v_t = tape.gradient(v, t)
        v_x = tape.gradient(v, x)
        v_y = tape.gradient(v, y)

        z_t = tape.gradient(z, t)
        z_x = tape.gradient(z, x)
        z_y = tape.gradient(z, y)

        # Define the physics-informed functions f_u, f_v, and f_c based on the gradients and equations.
        f_u = u_t + (u * u_x + v * u_y) + 9.81 * (z_x + z_y) + 0.0
        f_v = v_t + (u * v_x + v * v_y) + 9.81 * (z_x + z_y) + 0.0
        f_c = z_t + (u * z_x + v * z_y) + (h + z) * (u_x + v_y)

        return u, v, z, f_u, f_v, f_c


###############################################################################
# Main ########################################################################

if __name__ == "__main__": 

    N_train = 1000  # number of training points on domain.
    nIter = 50 #50000

    layers = [4, 20, 20, 20, 20, 20, 20, 20, 20, 3] # number of parameters/weights

    data = np.genfromtxt('../data/beach_2d.csv', delimiter=' ').astype(np.float32) # load data

    # Extract and flatten data.
    t_all = data[:, 0]                                          # Time, T x 1
    x_all, y_all, h_all = data[:, 1], data[:, 2], data[:, 3]    # Spatial coordinates and water depth, N x 1
    u_all, v_all, z_all = data[:, 5], data[:, 6], data[:, 4]    # Velocity and eta, N x T
    N, T = x_all.shape[0], t_all.shape[0]                       # Dimensions
    

    # Training ################################################################

    idx = np.random.choice(N, N_train, replace=False)
    t_train = t_all[idx][:, None]
    x_train, y_train, h_train = x_all[idx][:, None], y_all[idx][:, None], h_all[idx][:, None]
    u_train, v_train, z_train = u_all[idx][:, None], v_all[idx][:, None], z_all[idx][:, None]
    inputs_train = tf.concat([t_train, x_train, y_train, h_train], axis=1)
    outputs_train = tf.concat([u_train, v_train, z_train], axis=1)

    # Initializing and training the neural network model.
    model = PhysicsInformedNN(layers)

    start_time = time.time()   
    
    model.train(inputs_train, outputs_train, nIter)
    # Use L-BFGS-B for further optimization
    model.train_with_lbfgs(inputs_train, outputs_train)
    
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % elapsed)

    # Save the model
    model.save_weights('../log/model.ckpt')
    # to save entire model
    # model.save('./model')

    # Testing #################################################################
    # Setting up testing data for model evaluation.
    snap_idx = 2  # Replace this with the desired value of n
    start_r = snap_idx * 50
    end_r = (snap_idx + 1) * 50
    snap = np.arange(start_r, end_r)
    t_test = t_all[snap][:, None]
    x_test, y_test, h_test = x_all[snap][:, None], y_all[snap][:, None], h_all[snap][:, None]
    u_test, v_test, z_test = u_all[snap][:, None], v_all[snap][:, None], z_all[snap][:, None]
    inputs_test = tf.concat([t_test, x_test, y_test, h_test], axis=1)

    # Making predictions using the trained model.
    u_pred, v_pred, z_pred, f_u_pred, f_v_pred, f_c_pred = model.predict(inputs_test)

    # Save the testing locations and prediction results
    np.save('../log/x_test.npy', x_test)
    np.save('../log/y_test.npy', y_test)
    np.save('../log/u_pred.npy', u_pred)
    np.save('../log/v_pred.npy', v_pred)
    np.save('../log/z_pred.npy', z_pred)

    # Calculating errors between predictions and actual data.
    u_err, v_err, z_err = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2), np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2), np.linalg.norm(z_test-z_pred,2)/np.linalg.norm(z_test,2)
    print('Error u: %e' % u_err)
    print('Error v: %e' % v_err)
    print('Error z: %e' % z_err)
