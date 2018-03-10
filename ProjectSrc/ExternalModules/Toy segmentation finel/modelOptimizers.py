import tensorflow as tf

def optimizers(loss,learning_rate=0.001,type='sgd',adam_params)
    if type=='sgd:':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    if type=='Adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9
                                      ,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam').minimize(loss)
