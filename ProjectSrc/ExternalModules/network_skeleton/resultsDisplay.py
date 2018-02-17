import numpy as np
from matplotlib import pyplot as plt

def results_display(prediction=None,label=None,data=None):
    plt.figure(1)
    if prediction is not None:
        plt.subplot(311)
        plt.title('Prediction')
        plt.imshow(prediction, interpolation='nearest')
    if label is not None:
        plt.subplot(312)
        plt.title('Label')
        plt.imshow(label, interpolation='nearest')
    if data is not None:
        plt.subplot(313)
        plt.title('Image')
        plt.imshow(data, interpolation='nearest')
    plt.show()



