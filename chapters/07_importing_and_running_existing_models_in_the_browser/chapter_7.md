# Chapter 7 - Importing and Running Existing Models in TensorFlow.js Applications

In the last chapters we took at the workings of the TensorFlow.js Core API and became familiar with how to build models from scratch using the Layers API. In this chapter we will learn how to work with existing models that were created outside of the TensorFlow.js environment. After completing this chapter you should learn the following:

- How to save TensorFlow.js models for later use
- How to import existing models created using the Python version of TensorFlow or Keras
- How to convert saved TensorFlow.js models to Keras format
- How to retrain an imported model with new data

In this chapter we will be building a simple command-line application that will predict the price of a house provided some input. Rather than building the prediction engine ourselves we will import a pre-trained model. We will then accept user feedback to add new inputs to and retrain our model, which is very similar to how we would use TensorFlow.js in production.

## TensorFlow Deployment Strategy

As we learned in previous chapters, TensorFlow.js provides the ability to create and train models in the browser. This is a very powerful feature of the library and all of the models we’ve created so far were developed either in Node.js or the browser. However, training models can be both resource and time intensive. Developing deep learning models in the browser may not be the approach that we would want to take when building productions systems. Instead, we would want to develop our models outside of the browser using either TensorFlow (Python) or Keras (Python) or TFJS (JavaScript) like we did in previous chapters in this book. This way we can leverage the power of the user’s machine for development and that of a production server for retraining. Since most machine learning models must load a lot of data and will have to perform complex calculations on this data it makes sense to keep the development of our models offline and deploy the model to the browser for consumption.

## Saving Models

Before we talk about importing existing models into the browser we will briefly cover how to save models that we have developed using TensorFlow.js. In the previous chapter we saw how we were able to save the image recognition model we created in the previous chapter using the `model.save` method. Using this method we were able to save the model that we trained to a directory that lives on our computer. This works well if we are running a program on the server using Node.js; however, when performing work in the browser we do not have access to the user's file system. However, we do have alternative ways to save our models for future use.

When saving a model we use `model.save` we pass a string to the method as a parameter. This string is URL-like in nature and takes the following format: "<SCHEME>://<PATH>", where the scheme specifies where the model should be saved (i.e. to Local Storage or the file system in our example from chapter 6) and the path represents where the model should be saved. The call to this method is asynchronous so we treat like any other async method or function that returns a promise. When unwrapping the promise returned by this method call we receive a JSON object that contains information about the model including the model's weights and topology. We can call `model.save` on any `tf.Model` instance that

### Local Storage

It is possible to save our models directly to the browser by leveraging the Local Storage API. This API allows us to save data in the user's browser. Data stored here is read-only and does not expire and is saved across multiple browser sessions. This is useful if we are looking to retrain a model and save the retrained result to the user's browser. The scheme will be "localstorage://" and the path will be the name of our model, as demonstrated in the example below. We can then use the Local Storage API to retrieve and run our model.

```JavaScript
const savedModel = await model.save('localstoreage://model-to-save');
```

### IndexedDB

In addition to Local Storage, we have the option of saving our models to IndexedDB. IndexDB is a low-level API that is used for client-storage of data. The type of data that we can save to IndexedDB includes files and blobs (binary data). It is similar to Web Storage but was designed to enable high-performance searched of data stored to it and was designed to specifically store large amounts of data. If we are saving larger models it makes sense to save them to IndexedDB rather than Local Storage to take advantage of the storage efficiency and larger sized limit. Like saving the Local Storage the path will simply be the name of the file we wish to save. We can then use the IndexedDB API to retrieve and run our model.

```JavaScript
const savedModel = await model.save('indexeddb://model-to-save');
```

If we want to see what models we have saved to IndexedDB (as well as Local Storage) we leverage the `tf.io` API.

```javascript
const modelList = await tf.io.listModels();
console.log(modelList);
```

We can also copy our models between Local Storage and IndexedDB using `tf.io.copyModel`.

```javascript
tf.io.copyModel("localstorage://saved-model", "indexeddb://saved-model");
```

If wish to move a model rather than copy it from one location to another we can use `tf.io.moveModel`.

```JavaScript
tf.io.moveModel('localstorage://saved-model', 'indexeddb://saved-model');
```

Last, if we wanted to remove a model from either IndexedDB or Local Storage we can pass the scheme and path of the model we want to delete to `tf.io.removeModel`.

```javascript
tf.io.removeModel("indexeddb://saved-model");
```

### Downloading to the User's Desktop

Though we cannot save our models directly to the user's desktop we can initiate a download of the model as a file download. This will trigger a download of the model that will create a user experience similar to downloading a PDF or other file from the browser. The download result will be a JSON file named after the path that was specified to initialize the download (i.e. `model-to-save.json`).

```JavaScript
const savedModel = await model.save('download://model-to-save');
```

The resulting file will contain the model weights along with its topology. Along with this file comes a binary file that contains the weights of our model. This file is typically named as such: `model-to-save.weights.bin`.

### Server-side Saving

Saving newly created or retrained models in the user's browser is a useful feature but what if we wanted to save the model offline on our server somewhere? The good news is that TensorFlow.js allows us to save our models to a remote server using HTTP requests. This means to save our model we can do so via POST request. In order to gain more control over the request that is being sent we can use the `tf.io.browserHTTPRequest` API that allows us to specify the HTTP method and headers of our request.

```JavaScript
await model.save(tf.io-browserHTTPRequest('http://yourserver.com/upload', {method: 'PUT', headers: {'header_key': 'header_value'}}));
```

The body of this POST request will be in the `multipart/form-data` format, which is a common MIME format used for uploading files to servers. The body of the request will contain `model.json` and `model.weights.bin` files. These files are formatted identically to the files generated by the `downloads://` scheme.

### Saving to the User's File System

This is the strategy we used in the previous chapter and is available in tfjs-node. Since NOde.js can access the user's file system directly we are abel to use the node bindings to save our `tf.Model` instance to the user's computer. The only difference is that we must obviously be running our program that will be saving our model in a Node.js environment after importing `@tensorflow/tfjs-node`. Once this is done we can call `model.save` using the file system scheme and specifying a path to save our file to.

```JavaScript
import * as tf from '@tensorflow/tfjs-node';

await model.save('file:///trained-models/model-to-save');
```

This command will create a `model.json` and `weights.bin` files in a directory called `/trained-models/model-to-save`. These files will be of the same format as the files created when using the `download` and `http` schemes.

### Converting TensorFlow.js tf.Models into Keras Format

Once we have saved a `tf.Model` instance via the `downloads://` or `file://` schemas we can convert from TensorFlow.js format to in the format that Keras uses to load models (HDF5). To do this we will have to use teh tensorflowjs_converter that ships with the the `tensorflowjs` Python package. Once installed into a Python environment we can convert our file to HDF5 format using the following bash command.

```bash
tensorflowjs_converter --input_format tensorflowjs --output_format keras ./saved-model.json /trained_models/saved-model.h5
```

Once converted to this format we can load it into a Python program via the Keras library using its API methods. Keras is out of the scope of this book so it is suggested to read up on the Keras documentation to learn how to import external models for use with the library.

## Importing a Keras Model into TensorFlow.js

This section will demonstrate how to import a model created by Keras into a TensorFlow.js program. We will begin by creating a simple linear model in a Python program using Keras. We will then export this model so that we can import and use it in our TensorFlow.js program.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np


def main():
  # get our data from sklearn's make_blobs method
  X, y = make_blobs(n_samples=1000, centers=2, random_state=50)

  # split our model into training and testing portions
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  # create a sequential model, add a dense layer and train it
  model = Sequential()
  model.add(Dense(1, input_shape=(2,), activation='sigmoid'))
  model.compile(Adam(lr=0.05), loss='mse', metrics=['accuracy'])
  model.fit(X_train, y_train, epochs=100)

  # evaluate the model and print the results
  eval_result = model.evaluate(X_test, y_test)
  print('Loss: {}\nAccuracy: {}'.format(eval_result[0], eval_result[1]))

  # print the model summary
  model.summary()

  # save the model
  save_path = './output/linear_model'
  model.save(save_path)
  print('Model saved to: {}'.format(save_path))

  # make a sample prediction and print the results
  prediction = model.predict(np.array([[1, 2]]))
  print('Predicted result: {}'.format(prediction))

if __name__ == '__main__':
  main()
```

Here is a brief explanation of the program above. First we import our Keras dependencies from TensorFlow, namely the `Sequential` model, `Dense` layer and `Adam` optimizer. When then use a popular Python machine learning library, `scikit-learn`, to take advantage of its `Adam` optimizer, data sets and `train_test_split` method, which will split our data into training and testing data. We also use a library called Numpy that will make it easy for us to work with matrices and vectors.

Next we create our data set by making use of the `make_blobs` method imported from scikit-learn. Next we split this data into a training to be used for training and test set, which we will use for evaluating our model after testing. Next we create a `Sequential` model and add a `Dense` layer to it, defining its shape and specifying an activation function to use. Next we compile our model using the `Adam` optimizer with a learning rate 0.05 while minimizing the mean squared error (MSE) cost function and evaluating the accuracy of the model. After compiling our model we train it using the `model.fit` method, which we pass our training data. we train our model for 100 epochs. After the training step is complete we evaluate the model and print the results. We then print the model summary provided by Keras, save the model for reuse later and then perform inference on the model by passing it a data point and printing the result.

By examining the code above, it becomes clearer how the TFJS Layers API closely resembles the API provided by Keras. Also by looking at the example we can see that Keras ships with the Python version of TensorFlow by default. Let's continue by importing the the Keras model we saved in our Python into a JavaScript program using TensorFlow.js

TODO: IMPORT MODEL AND PERFORM INFERENCE

```javascript
```

TODO: FINISH THIS!!!

### Importing a Python TensorFlow Model into TensorFlow.js

In this section we will demonstrate how to import a saved TensorFlow model, created using the Python version of TensorFlow, into a TensorFlow.js program. We will be using the output created by the following Python program, which creates and trains a model that simply multiplies a list of inputs by a factor of 100. Don't worry too much about the Python specifics in the file. It's simply provided to demonstrate the similarities between the Python TensorFlow library and the JavaScript/Node version as well as to provide insight into what the model we're importing is doing.

```python
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
```

As a brief explanation, this program will simply create a vector of inputs as an `np.ndarray` from 1 to seven in steps of 0.5 increments and multiple each item by 100 to get our outputs. When then set a random weight and bias term to use for our regression formula. Next we define our cost function (MSE) and use an instance of `tf.train.GradientDecentOptimizer` to use Gradient Descent to minimize our cost function using a learning rate 0.05. Next we initialize our session variables, instantiate an instance of `tf.train.Saver`, which save our model result. For our last step we train our model for our number of epochs, print the results every 100 epoch. Finally we save the model result and then test the model by making a prediction by feeding a value, 2 in this case, to our model to see if it can correctly predict the result. The predicted result is 200.0001, which is extremely close to 2 \* 100, which is essentially what our model our model is expected to return so we can be comfortable that it is working correctly.

Though this example is uses the Python version of TensorFlow, we can see similarities between this code and the code we have written in previous chapters using tfjs-node. The next step is to import it into a JavaScript program and use it to perform inference. We can do this as follows.

TODO: IMPORT MODEL AND PERFORM INFERENCE

```javascript
```

TODO: FINISH THIS SECTION!!!

## Transfer Learning

Transfer learning is the process of taking a model that is trained to perform a particular task is reused to perform a similar or totally different task. It is a popular machine learning technique that is used to take a pre-trained model that was trained for a specific use case to apply it to another use case. For example, say we wanted to use a model that was developed to recommend movies to a user but we wanted to use the model to recommend books.

Another benefit of using transfer learning is the time it takes to train the model. For example, a model like Google DeepMind's Inception takes several weeks to train. If we wanted to use this model without having to wait several weeks to train the model we can use transfer learning to use the model and only make a change to one of the layers to fit out needs. Taking this approach we can take models that trained on a specific use case and tailor for our specific use case.

The TensorFlow.js library was build with transfer learning in mind. It allows us to pre-load existing models in the browser and allows us to easily retrain these models using new input that is provided by the user without having to make a round-trip to the server. In the next section we'll see this in action.

## Retraining Existing Models with User Input

TBD - code along tutorial pending changes to code examples

For this chapter's tutorial we will install a housing prediction model from TensorFlow Hub. 

## Summary

In this chapter we learned how to import existing models for use in our browser-based applications. We also learned how to use these imported models to perform inference on user inputs that were fed into a simple price prediction engine. Next we reviewed the steps required to load models developed in Keras or the Python and C++ TensorFlow libraries into the browser using the `tf.loadModel` method. Last we discovered how we can refrain the models we imported into the browser with new data points collected from the user. In the next chapter we will focus on deployment and how e can write our code more efficiently for production use.
