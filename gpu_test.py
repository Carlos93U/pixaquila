import tensorflow as tf
import time

print('*'*50)
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.test.is_gpu_available())
print("Devices available:", tf.config.list_physical_devices())
print('*'*50)

# Genera un tensor grande
matrix_size = 10000
with tf.device('/GPU:0'):
    print("Running on GPU...")
    start = time.time()
    a = tf.random.normal([matrix_size, matrix_size])
    b = tf.random.normal([matrix_size, matrix_size])
    c = tf.matmul(a, b)
    print("Operation completed on GPU in:", time.time() - start, "seconds")
