import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def results_display(logits, label, data, index, rindex, thresh, imageSize):
    prediction=sigmoid(logits)
    prediction[prediction>thresh]=1
    prediction[prediction<=thresh]=0
    plt.figure(1)
    plt.subplot(311)
    plt.title('Label')
    plt.imshow(np.reshape(label[index], [imageSize, imageSize]), cmap='gray')
    plt.subplot(312)
    plt.title('Prediction')
    plt.imshow(np.reshape(prediction[rindex], [imageSize, imageSize]), cmap='gray')
    plt.subplot(313)
    plt.title('Image')
    plt.imshow(np.reshape(data[index],[imageSize,imageSize]),cmap='gray')
    plt.show()





