from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    Path('plots').mkdir(parents=True, exist_ok=True)
    x1 = np.arange(-8, 8, 0.1)
    x2 = np.arange(-4, 4, 0.1)
    sigmoid = tf.sigmoid(x2)
    tanh = tf.tanh(x2)
    relu = np.maximum(0, x1)
    relu6 = np.minimum(np.maximum(0, x1), 6)
    swish = x1 * tf.sigmoid(x1)
    hswish = x1 * np.minimum(np.maximum(0, x1+3), 6) * (1/6)

    fig1 = plt.figure()
    plt.plot(x2, sigmoid, label='sigmoid')
    plt.plot(x2, tanh, label='tanh')
    plt.legend()
    plt.grid()
    plt.title('sigmoid vs tanh')
    plt.savefig('plots/sigmoid_tanh.png')

    fig2 = plt.figure()
    plt.plot(x1, swish, '--', label='swish')
    plt.plot(x1, relu6, label='ReLU6')
    plt.plot(x1, hswish, '-', label='hswish')
    plt.legend()
    plt.grid()
    plt.title('ReLU6 vs Swish vs Hard-Swish')
    plt.savefig('plots/relu6_swish_hswish.png')
