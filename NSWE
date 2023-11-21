"""
PINN for reconstructing UV field with nonlinear shallow water equations and lab data
@author: Reza Salatin
Nov 21, 2023
"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


np.random.seed(1234)
tf.set_random_seed(1234)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
################################ PINN ################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# 
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, h, t, u, v, z, layers):
        
        X = np.concatenate([x, y, h, t], 1)

        self.X = X
        
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.h = X[:,2:3]
        self.t = X[:,3:4]

        self.u = u
        self.v = v
        self.z = z
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.h_tf = tf.placeholder(tf.float32, shape=[None, self.h.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])

        # tf Graphs
        self.u_pred, self.v_pred, self.z_pred, self.f_u_pred, self.f_v_pred, self.f_c_pred = self.net_NSWE(self.x_tf, self.y_tf, self.h_tf, self.t_tf)
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.v_tf - self.v_pred)) + \
                    tf.reduce_mean(tf.square(self.z_tf - self.z_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_v_pred)) + \
                    tf.reduce_mean(tf.square(self.f_c_pred))
        
        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
              

    ######################## NN Initialization ##########################
    def initialize_NN(self, layers):       

        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        

    ######################## Xavier Initialization ##########################
    def xavier_init(self, size):

        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    

    ############################## NN ##########################
    def neural_net(self, X, weights, biases):

        num_layers = len(weights) + 1

        #H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 #normalization of input data
        H = X

        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    

    ############################## Physics ##########################
    def net_NSWE(self, x, y, h, t):
        
        uve = self.neural_net(tf.concat([x,y,h,t], 1), self.weights, self.biases)
        u = uve[:,0:1]
        v = uve[:,1:2]
        e = uve[:,2:3]
        
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        
        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        
        z_t = tf.gradients(z, t)[0]
        z_x = tf.gradients(z, x)[0]
        z_y = tf.gradients(z, y)[0]

        f_u = u_t + (u*u_x + v*u_y) + 9.81 * (z_x+z_y)
        f_v = v_t + (u*v_x + v*v_y) + 9.81 * (z_x+z_y)
        f_c = z_t + (u*z_x + v*z_y) + (h+z) * (u_x+v_y)

        return u, v, e, f_u, f_v, f_c
    

    def callback(self, loss):

        print('Loss:', loss)
        

    ############################## Training ##########################
    def train(self, nIter):
        
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, 
                   self.t_tf: self.t,
                   self.u_tf: self.u, self.v_tf: self.v,
                   self.e_tf: self.e}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        


    ############################## Predict ##########################
    def predict(self, X_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        z_star = self.sess.run(self.e_pred, tf_dict)
        
        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)
        f_c_star = self.sess.run(self.f_c_pred, tf_dict)
               
        return u_star, v_star, z_star, f_u_star, f_v_star, f_c_star
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
################################ Main ################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# 

if __name__ == "__main__": 
    
    # Number of training points on domain
    N_train = 500
    
    # 4 inputs (x,y,h,t), 8 layers, 20 nodes, 3 outputs (u,v,e)
    layers = [4, 20, 20, 20, 20, 20, 20, 20, 20, 3]
    
    # Load data from .mat file
    data = scipy.io.loadmat('../Data/NSWE.mat')
    
    # Data                              # Size 
    U_star = data['UV'] # (u,v)         # N x 2 x T  
    Z_star = data['Z'] # (eta)          # N x T     

    t_star = data['t'] # (time)         # T x 1
    X_star = data['XY'] # (X,Y)         # N x 2 
    h_star = data['h'] # (h)            # N x 1 
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T))  # N x T
    YY = np.tile(X_star[:,1:2], (1,T))  # N x T
    hh = np.tile(h_star, (1,T))         # N x T
    TT = np.tile(t_star, (1,N)).T       # N x T
    
    UU = U_star[:,0,:]                  # N x T
    VV = U_star[:,1,:]                  # N x T
    ZZ = Z_star                         # N x T
    
    # Flatten all data to randomly pick from
    x = XX.flatten()[:,None]            # NT x 1
    y = YY.flatten()[:,None]            # NT x 1
    h = hh.flatten()[:,None]            # NT x 1
    t = TT.flatten()[:,None]            # NT x 1
    
    u = UU.flatten()[:,None]            # NT x 1
    v = VV.flatten()[:,None]            # NT x 1
    z = ZZ.flatten()[:,None]            # NT x 1
    

    ############################ Training Data ###########################
    # Training Data (Random pick N_train points)
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    h_train = h[idx,:]
    t_train = t[idx,:]

    u_train = u[idx,:]
    v_train = v[idx,:]
    z_train = z[idx,:]
            
    # Training
    model = PhysicsInformedNN(x_train, y_train, h_train, t_train, u_train, v_train, z_train, layers)
    start_time = time.time()   
    model.train(50000)
    elapsed = time.time() - start_time                    
    print('Training time: %.4f' % (elapsed))


    ############################## Testing Data ##########################
    # Testing Data (one snapshot)
    snap = np.array([100])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    h_star = h_star
    t_star = TT[:,snap]

    u_star = U_star[:,0,snap]
    v_star = U_star[:,1,snap]
    z_star = Z_star[:,snap]
    
    # Prediction
    u_pred, v_pred, z_pred, f_u_pred, f_v_pred, f_c_pred = model.predict(x_star, y_star, h_star, t_star)
            
    # Error Calculation
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_z = np.linalg.norm(z_star-z_pred,2)/np.linalg.norm(z_star,2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error e: %e' % (error_z))












'''
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    ############################# Plotting ###############################
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')
    
    ####### Row 0: h(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
#    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$|h(t,x)|$', fontsize = 10)
    
    ####### Row 1: h(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact_h[:,75], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')    
    ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])    
    ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
    
    # savefig('./figures/NLS')  
    
'''
