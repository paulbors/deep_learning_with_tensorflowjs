# Chapter 4 - Your First Artificial Neural Network

In the previous chapter we learned about the theory that drives artificial neural networks. In this chapter we will build an artificial neural from scratch using TensorFlow.js. It will act as a simple tutorial that will demonstrate how TensorFlow.js can be used to create an artificial neural network that recognizes handwritten digits.

To accomplish this we will cover the following topics:

- Understand how TensorFlow.js can be used to create artificial neural networks How to load existing data sets
- Understand how data flows through a neural net
- Understand the different types of tensors
- Learn about the different tensor operations
- How to train a model in the browser
- Understand how sessions are used to train models Learn how to assess the strength of a model

## TensorFlow.js

TensorFlow.js is an open-source library that allows users to create, train and run deep learning models directly in the browser using JavaScript as a language. This provides the ability to run machine learning models entirely on the client-side without having to make embedded in web applications that are served over the web without the user having to install any libraries or drivers for the software to work. Because Tensorflow.js runs in the browser it targets the user’s GPU via WebGL. This means that if WebGL is available TensforFlow.js will automatically detect the GPU and use it accelerate the performance of the model.

Other benefits of TensorFlow.js are listed below:

- TensorFlow.js supports the use of pre-existing models - Existing models created using the TensorFlow Python or Keras libraries can be imported into the browser for consumption by TensforFlow.js.
- Models can be trained directly in the browser and exported for future use - This makes is simple to use models developed using the TensorFlow Python API on the client.
- TensorFlow.js has an API that is similar to the Python API
- The TensforFlow.js API has an API that is similar to the TensorFlow Python API, however, not all functionality of the TensorFlow Python API is supported by TF.js out of the box. However, this may change in the near future as the TensforFlow team is working to achieve API parity between the two libraries.
- The performance of TensorFlow.js is not that far off from the Python API for smaller models - As noted in a previous chapter, at the time of this writing, TensorFlow.js with WebGL is roughly 1.5 - 2x slower than TensorFlow Python with AVX. Google notes that they see small models actually train faster in the browser while larger models with lots of data can take up to 10-15x slower in the browser when compared to Python with AVX.
- The TensorFlow.js Layers API is similar to the Keras Python library

TensorFlow.js can also be used on the client via the node.js bindings. The node.js bindings allow the same JavaScript code that works in the browser to work on the server leveraging the node.js platform. This is accomplished by binding to the TensorFlow C implementation. This can be installed via NPM with a simple `npm i @tensorflow/tfjs-node` command.

The TensorFlow.js library provides two APIs that can be used to build, train and use deep learning models:

- Layers API - provides a high-level API that allows users to easily build an artificial neural network. The API also provides methods that allow us to create different kinds of layers and apply activation functions and optimization techniques. This API is built on top of the lower level Core API.
- Core API - provides a low-level API that can be used to develop machine learning models. It is composed of tensors, variables and operations that are run in a session for training and inference.

In this chapter we will use TensorFlow.js’ layers API to build an artificial neural network that can recognize images -- particularly handwritten digits.

## Deep Learning in the Browser

To build our first artificial neural network using TensorFlow.js we will continue with our hello world React example from chapter two. We will build our image recognition model in the body of our runModel function. Start by clearing out the body of this function. Your App.js file should look as follows:

**Figure 4.1**

```javascript
// src/App.js

import React, { Component } from "react";
import * as tf from "@tensorflow/tfjs";

class App extends Component {
  runModel = () => {
    // TODO: add code
  };

  render() {
    this.runModel();
    return (
      <div className="App">
        <h1>Tensorflow.js Quick Start Guide</h1>
        <p>Check the console for model result.</p>
      </div>
    );
  }
}

export default App;
```

## Creating Our First Layers

Layers are the workhorses of of deep learning models. The job of each layer is to receive data as its input, perform some operation on that data and return the result of that operation as its output. This is achieved through use of the Core API. When we create a layer TensorFlow.js will automatically create and initialize the values required to create, train and perform inference of on the artificial neural network that we will build. In the next section we will create simple feed forward neural network levaring the Layers API.

**Figure 4.2**

```javascript
// App.js
import React, { Component } from "react";
import * as tf from "@tensorflow/tfjs";

class App extends Component {
  runModel = () => {
    // TODO: add code
  };
  render() {
    this.runModel();
    return (
      <div className="App">
        <h1>Tensorflow.js Quick Start Guide</h1>
        <p>Check the console for model result.</p>
      </div>
    );
  }
}
export default App;
```

The first thing we’ll want to do is define our model. TensorFlow.js allows us to easily do this in one line using the `tf.sequential` method. The `tf.sequential` method returns a `tf.Sequential` model instance. This type of model allows us to create a model where the outputs of layer are used asn inputs to the next layer. This creates a “stack” of layers. This is easily done by adding the following line to the top of our runModel function:

**Figure 4.3**

```javascript
// App.js
const model = tf.sequential();
```

Next we’ll have to configure our model. We achieve this by adding layers and defining their dimensions. This done with the “add” method provided by the `tf.Sequential` model instance we created in the previous step.

**Figure 4.4**

```javascript
// App.js
model.add(
  tf.layers.dense({
    inputShape: [3],
    activation: "sigmoid",
    units: 4
  })
);
```

In the above code we add a new dense layer via the tf.layers.dense method. The tf.layers namespace provides a set of operations and weights taht can be used to create a model. We use tf.layers.dense to create a new dense layers that has a shape of 3, uses a sigmoid activation function and consists of four units. Next we’ll define the next layer that this layer will pass its output to.

**Figure 4.5**

```javascript
model.add(
  tf.layers.dense({
    units: 2,
    activation: "sigmoid"
  })
);
```

We now have two layers in our artificial neural network. The first takes some data as its input and performs some operation on this data and passes it off the second dense layer that we created in figure 4.6. This layer will serve as the final layer to our model and will be referred to as our “output” layer. Next we’ll define the optimization algorithm and cost function that we want to minimize.

**Figure 4.6**

```javascript
// App.js

const optimizer = tf.train.sgd(0.1);
```

In figure 4.7 we create a Stochastic Gradient Descent optimizer. In chapter 3 we learned how this optimizer works by solving for the minimum of a loss function. Or in other words, we are looking for the hyperparameter values in our models (essentially our weights and biases) that provide the smallest amount of errors in our model output. By calling `tf.train.sdg` we are returned a `tf.SGDOptmizer` object that will perform the Stochastic Gradient Descent algorithm we learned about in chapter 3 programmatically when we are ready to perform our learning step. The parameter that we pass to the `tf.train.sgd` constructor function is the learning rate that we want to use to train our model.This value is also referred to as our “alpha” value and represents the size of the steps we will take in either direction as we attempt to find the global minimum of our chosen loss function, which we will select in our next step.

**Figure 4.7**

```javascript
// App.js
model.compile({
  optimizer,
  loss: "meanSquaredError"
});
```

Next we call the compile method on our model instance. The compile method configures and prepares our model for its training and evaluation steps. It takes a configuration object, which specifies and optimizer (in our case we chose Stochastic Gradient Descent) and the loss function to apply our optimization algorithm to. In our case we chose to use the Mean Squared Error (MSE), which we learned about in chapter 3, as our loss function. With this in place we are now ready to gather our data and train our model.

**Figure 4.8**

```javascript
// App.js

// input/feature data
const x_train = tf.tensor([
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
  [0.7, 0.8, 0.9],
  [0.9, 0.8, 0.7]
]);
// training labels
const y_train = tf.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]);
```

In figure 4.9 we create out training data and labels for this data. The training data can be thought of as our model features. The labels are classifications for this data. The job of our artificial neural network is to learn the pattern between the features and labels so that when it sees new feature data as its input it can correctly predict the best label that matches that data. Next we’ll code up our training steps.

**Figure 4.9**

```javascript
// App.js
// model training function
const trainModel = async (epoch = 1000, batch_size = 10) => {
  for (let i = 0; i < 100; i++) {
    const result = await model.fit(x_train, y_train, epoch, batch_size);
    console.log(result.history.loss[0]);
  }
};
```

Above we create a function called trainModel that specifies the number of epochs and batch size to use when training our model. As we learned in chapter 3, an epoch is when an entire data set is passed forward and backward through the neural network one time. Since an epoch can be large we usually split them up into batches. Batches are the number of training examples that we want to include in a bacth. The number of iterations is how many times we will fit our model to our training data. Once we have completed all iterations we have successfully trained our model. In the example above, we log the resulting value of our loss function, which would expect to decrease with each iteration. As the last step we will make predictions from our trained model.

**Figure 4.12**

```javascript
// Test data
const test_data = tf.tensor([[0.1, 0.2, 0.3]]);
trainModel(100, 1000, 10).then(() => {
  console.log("Model training complete: ");
  model.predict(test_data).print();
});
```

By adding the last code snippet we first train our model for 100 iterations, with an epoch of 1000 and a batch size of 10. After training is complete we pass our test data and receive a tensor as our output prediction.

**Figure 4.13**

```javascript
Tensor
[[0.4555036, 0.5442941],]
```

Congratulations! You have successfully created a simple feed forward artificial neural network using the TensorFlow.js Layers API. This example demonstrates how we can build an train a simple ANN using random features and labels.

## Summary

In this chapter we learned how to create a simple feedforward artificial neural network to predict labels for random data that we generated. We then used the TensorFlow.js to load a data set called MNIST, which consists of 60,000 training examples and 10,000 test examples of handwritten digits from 0 to 9. We then used the TensorFlow.js Layers API to build a convolutional neural network consisting of convolution, pooling and dense layers to recognize and classify handwritten digits it has never seen before.

In the next chapter we will take a deep dive into the TensorFlow.js Core API and learn how its basic building blocks are used to create the Layers API that we used to train a simple image classifier using the MNIST data set. We will also learn more about memory management and how we can optimize our model runs by running them in sessions.
