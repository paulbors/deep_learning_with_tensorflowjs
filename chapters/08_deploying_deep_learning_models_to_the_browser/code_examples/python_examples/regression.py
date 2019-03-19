import tensorflow as tf
import pandas as pd 
import numpy as np 


def main():
  # set our learning rate and number of epochs for use later
  learning_rate = 0.05
  epochs = 2000

  # create our training data
  train_X = np.arange(1, 7, .5)
  train_Y = train_X * 100

  # get the number of samples
  n_samples = len(train_X)

  # create placeholders for our inputs and outputs
  X = tf.placeholder('float')
  Y = tf.placeholder('float')

  # define model weights and bias term
  W = tf.Variable(np.random.randn(), name='weight')
  b = tf.Variable(np.random.randn(), name='bias')

  # linear regression formula y^ = b + X*W
  y_hat = tf.add(tf.multiply(X, W), b)

  # MSE
  cost = tf.reduce_sum(tf.pow(y_hat - Y, 2)) / (2 * n_samples)
  
  # create our gradient descent optimizer
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  # initialize our variables
  init = tf.global_variables_initializer()

  # create saver 
  saver = tf.train.Saver()

  # train the model
  with tf.Session() as sess:
      sess.run(init)

      # fit the data to the model
      for epoch in range(epochs):
          for (x, y) in zip(train_X, train_Y):
              sess.run(optimizer, feed_dict={X: x, Y: y})

          # log results on every 100th iteration
          if (epoch + 1) % 100 == 0:
              c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
              print('Epoch: {}, Cost: {}, W: {}, b: {}'.format(epoch+1, c, sess.run(W), sess.run(b)))
      
      print('Training complete!')

      training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})

      print('Training cost: ', training_cost, 'W: ', sess.run(W), 'b: ', sess.run(b), '\n')

      # save model
      save_path = saver.save(sess, './output/regression.ckpt', global_step=epochs)
      print('Model saved to: {}'.format(save_path))

      # perform inference (should return twice the input)
      prediction = sess.run(y_hat, feed_dict={X: [2]})
      print('Predicted result: {}'.format(prediction))

if __name__ == '__main__':
  main()
