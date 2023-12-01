import tensorflow as tf
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("CUDA version:", tf.version.COMPILER_VERSION)
