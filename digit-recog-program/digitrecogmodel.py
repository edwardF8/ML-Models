import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

model = tf.keras.Sequential([tf.keras.Dense(units=3), activation='sigmoid'),
                            