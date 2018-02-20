import tensorflow as tf

def diceScore(logits, labels):
    eps = 1e-5

    prediction = tf.round(tf.nn.sigmoid(logits))
    intersection = tf.reduce_sum(tf.multiply(prediction, labels))
    union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(labels)
    res = 2 * intersection / (union + eps)
    return res.eval()


def accuracy(logits, labels):
    predictions = tf.round(tf.nn.sigmoid(logits))
    eq = tf.equal(predictions, labels)
    res = tf.reduce_mean(tf.cast(eq, tf.float32))
    return res.eval()