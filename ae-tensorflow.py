import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# importing the datasets
movies = pd.read_csv("ml-1m/movies.dat", sep = '::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv("ml-1m/users.dat", sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv("ml-1m/ratings.dat", sep = '::', header=None, engine='python', encoding='latin-1')

# preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = training_set[np.sum(training_set, axis=1)>0]
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

def compute_loss(predictions, labels,num_labels):
    with tf.name_scope('loss'):
        loss_op=tf.sqrt(tf.div(tf.reduce_sum(tf.square(tf.subtract(predictions,labels))),num_labels+1e-10))
    return loss_op
     
def create_autoencoder_model(n_inputs, X):
    # encoder
    n_hidden1 = 20
    n_hidden2 = 20
    n_hidden3 = 20
    activation = tf.nn.sigmoid
    regularizer = None #tf.contrib.layers.l2_regularizer(0.001)
    initializer = tf.contrib.layers.variance_scaling_initializer()
    w1 = tf.get_variable(shape=[n_inputs, n_hidden1], dtype=tf.float32, initializer = initializer, regularizer = regularizer, name='w1')
    b1 = tf.get_variable(shape=[n_hidden1], dtype=tf.float32, initializer = initializer, regularizer = regularizer,name='b1')
    w2 = tf.get_variable(shape=[n_hidden1, n_hidden2], dtype=tf.float32, initializer = initializer, regularizer = regularizer,name='w2')
    b2 = tf.get_variable(shape=[n_hidden2], dtype=tf.float32,initializer = initializer, regularizer = regularizer, name='b2')
    w3 = tf.get_variable(shape=[n_hidden2, n_hidden3], dtype=tf.float32, initializer = initializer, regularizer = regularizer,name='w3')
    b3 = tf.get_variable(shape=[n_hidden3], dtype=tf.float32,initializer = initializer, regularizer = regularizer, name='b3')
    w4 = tf.get_variable(shape=[n_hidden3, n_inputs], dtype=tf.float32, initializer = initializer, regularizer = regularizer,name='w4')
    b4 = tf.get_variable(shape=[n_inputs], dtype=tf.float32, initializer = initializer, regularizer = regularizer,name='b4')
    hidden1 = activation(tf.matmul(X,w1) + b1)
    #dropout1 = tf.nn.dropout(hidden1, keep_prob=0.7)
    hidden2 = activation(tf.matmul(hidden1,w2)+b2)
    dropout2 = tf.nn.dropout(hidden2, keep_prob=0.5)
    hidden3 = activation(tf.matmul(dropout2,w3)+b3)
    #dropout3 = tf.nn.dropout(hidden3, 0.7)
    output = tf.matmul(hidden3, w4)+b4
    #mask=tf.where(tf.equal(X,0.0), tf.zeros_like(X), X) # indices of zero values in the training set (no ratings)
    num_train_labels=tf.cast(tf.count_nonzero(X),dtype=tf.float32) # number of non zero values in the training set
    bool_mask=tf.cast(X,dtype=tf.bool) # boolean mask
    masked_output=tf.where(bool_mask, output, tf.zeros_like(output)) # set the output values to zero if corresponding input values are zero
    reconstruction_loss= compute_loss(masked_output,X,num_train_labels)
    #output = tf.matmul(hidden3, w4)+b4
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([reconstruction_loss]+reg_loss)
    return output, loss

tf.reset_default_graph()
X = tf.placeholder(dtype=tf.float32, shape=[None, nb_movies])
output, loss = create_autoencoder_model(nb_movies,X)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9, momentum= 0.1)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

batch_size = 1
epochs = 500
def gen_inputs(data, batch_size):
    for i in range(0, len(data)//batch_size):
        start = i*batch_size
        end = np.min([start+batch_size,len(data)])
        X = np.array(data[start:end])
        X = X[np.sum(X>0, axis=1)>0]
        yield X

saver = tf.train.Saver()
with tf.Session() as sess:
    try:
        saver.restore(sess, './model.chkp')
    except Exception:
        sess.run(init)
    for epoch in range(epochs):
        for data in gen_inputs(training_set, batch_size):
            _, err = sess.run([training_op, loss], feed_dict={X:data})
        print("epoch: {}, loss: {}".format(epoch, err))
    saver.save(sess,'./model.chkp')

# test the result model
test_lost = 0.
s=0.
with tf.Session() as sess:
    saver.restore(sess,'./model.chkp')
    for user_id in range(nb_users):
        inp = np.array(training_set)[[user_id]]
        target = np.array(test_set)[[user_id]]
        positive_rating = np.sum(target>0, axis=1)[0]
        if positive_rating>0:
            predictions = sess.run(output, feed_dict={X:inp})
            predictions[target==0]=0
            #mean_corrector = nb_movies/float(positive_rating+1e-10)
            #err = sqrt(mean_squared_error(target, predictions)*mean_corrector)
            err = compute_loss(predictions, target, positive_rating)
            test_lost+=sess.run(err)
            s+=1

print("test loss:", test_lost/s)
    

