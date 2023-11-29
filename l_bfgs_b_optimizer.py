import tensorflow as tf
import numpy as np
from scipy.optimize import minimize

class LBFGSBOptimizer:
    def __init__(self, model, inputs, outputs, loss_function, options=None):
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.loss_function = loss_function
        self.options = options or {'maxiter': 50000, 'maxfun': 50000, 'maxcor': 50, 'maxls': 50, 'ftol' : 1.0 * np.finfo(float).eps}
        self.shapes_and_sizes = [(weight.shape, tf.size(weight).numpy()) for weight in model.trainable_weights]

    def function_for_scipy(self, x):
        # Convert flat x back into the list of weight arrays
        weights = []
        start = 0
        for shape, size in self.shapes_and_sizes:
            weights.append(tf.reshape(x[start:start + size], shape))
            start += size

        self.model.set_weights(weights)

        with tf.GradientTape() as tape:
            loss = self.loss_function(self.model, self.inputs, self.outputs)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
        return loss.numpy(), gradients.numpy()

    def optimize(self):
        x0 = np.concatenate([weight.numpy().flatten() for weight in self.model.trainable_weights])
        result = minimize(fun=self.function_for_scipy, x0=x0, jac=True, method='L-BFGS-B', options=self.options)
        self.model.set_weights(result.x)
