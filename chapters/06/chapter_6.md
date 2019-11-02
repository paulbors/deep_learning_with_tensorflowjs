# Chapter 6: Building Artificial Neural Networks with the TensorFlow.js Layers API

In chapter four we built our very first Artificial Neural Network (ANN) using the TensorFlow.js layers API. In that chapter we skimmed over the API and focused on the process involved in creating a deep learning model as demonstration of the power that TensorFlow.js provides when it comes to building Artificial Neural Networks. In this chapter we will take a deeper dive into this API to learn how it is structured as well as what methods and data structures it provides to allow us to create deep learning models that can be run in the browser. By the end of this chapter we will have accomplished:

- Learning how to build an ANN network from layers
- Understanding the different types of layers
- How to use the Model and Sequential layer types
- Understand how to choose and use activations
- Understand how to reshape data for analysis and model training
- How to compute and optimize a loss function

> Note: To keep the focus on the API an its methods we will be using the `tfjs-node` project. It's important to note that at the time of this writing this API is experimental and is not ready for production use. However, since TensorFlow is coming to Node.js it is worth gaining on understanding of how TensorFlow can be used in a Node environment. Since this chapter focuses on the TFJS API methods, which are the same between the `@tensorflow/tfjs` and `@tfjs-node` packages the same examples that are presented here should also work in the browser. Further instructions on how to run these code examples can be found in the code samples directory.

## Layers

In the previous chapter we focused on the power provided to us by the TensorFlow.js Core API. In this chapter we will go a little higher level and take a dive into the Layers API and how it can help us quickly build Artificial Neural Networks. The Layers API provides a higher level interface to work when building our artificial neural networks. It is modeled after the Keras library, which we learn a little about in the next section.

### A Little Bit About Keras

Keras is a deep learning library that was original written for Python. It acts as an interface that wraps several deep learning backends, including but limited to just TensorFlow. Keras is popular in the deep learning community and has been made the official high-level library of TensorFlow and is actually packaged with the Python version of TensorFlow.

Keras provides a simple API that allows us to create layers that either represent our input data, transformation on our data or the result of our overall model. These layers are created by creating an instance of a Sequential class. Once an instance is created we can use the Keras API to add additional layers, apply some transformation to the layer, add activation functions or compile our model in just a few lines. We can train our model and assess its strength. We’ll see that the TensorFlow.js Layers API works in a very similar fashion. In fact, Google has written the Layers API to model Keras as much as possible though there are slight differences in the API due to language specific differences between JavaScript and Python.

### Types of Layers

Layers are primary structure used to build models in TensorFlow.js when using the Layers API. Each layer is typically used to perform a computation to transform its input into some output than is either fed into another layer or used as output for analysis.

### tf.sequential

To create a sequential model we can use the tf.sequential factory function to create a new tf.Sequential model. A sequential model is defined by being a model where the outputs of one layer are used as input to the next. This essentially allows us to build a deep learning model as a stack of layers where each layer performs a transformation on its inputs and the results are used an input to the next layer.

**Figure 6.1**

```javascript
const model = tf.sequential();
model.add(tf.layers.dense({ units: 32, inputShape: [100] }));
model.add(tf.layers.dense({ units: 4 }));
```

In the example above we call the `tf.sequential()` factory function to create a new sequential model instance. We then add two dense layers to this model.

### tf.Sequential

In the previous section we learned that a `tf.Sequential` model is a special kind of model created from a “stack” of layers where the output of of one layer linearly feeds to the next layer in the model. In this section we will take a closer look at the `tf.Sequential` class that represents this kind of model.

The `tf.Sequential` class actually extends the `tf.Model` class. The `tf.Model` class represents the basic unit of training and can be created using the `tf.model()` factory function. The difference between `tf.model()` and `tf.sequential()` is that `tf.model()` is generic and supports an arbitrary graph of layers, whereas `tf.sequential` is less generic and supports a specific type of model, which is created by stacking layers. Since we’ll mostly be working with sequential models we will keep the focus on the `tf.Sequential` model; however, it is important to know that `tf.Model` exists especially since `tf.Sequential` is an extension of this class.

In figure 6.2 we’ll see how we can create a simple linear regression model using the `tf.Sequential` class and its methods.

**Figure 6.2**

```javascript
const predict = async () => {
  // create model and add a dense layer
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // compile the model
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  // traning data
  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

  // train model with our training data
  await model.fit(xs, ys);

  // use the model to predict an output
  model.predict(tf.tensor2d([5], [1, 1])).print();
};

// invoke our predict function
predict();
```

In the example above we create a new sequential model by invoking the `tf.sequential()` factory function. This returns a `tf.Sequential` class instance that we can use to build our sequential model. The `tf.Sequential` class provides and add method that we use to add dense layer using the `tf.layers.dense` factory method. We specify that this layer should consist of unit and has one element in one dimension. Next we compile our model, which pares it for training and evaluation by specifying our loss function and optimization algorithm (respectively MSE and stochastic gradient descent in our example). Next we prepare some sample training data. The `xs` label represents our observations while the `ys` label represent expected outputs. We take this data and fit it the sequential model that we created in our first step and then use this trained model to predict a label for data our model has never seen before. Though it was a very trivial one we can see how powerful the Layers API is in that it allowed us to create and train a model in only seven lines of code!

### tf.Sequential API Overview

As we were able to learn in the last section, the `tf.Sequential` class makes it really simple for us to create, train and evaluate a model in very few lines of code. The clean syntax and API methods also makes it very easy to follow the flow of model and what is happening at each step. In this section we will explore the features of the `tf.Sequential` class:

- **compile**​ - this method configures and prepares our model for the training and evaluation steps. When we compile a model we specify an optimizer, loss function and any ​metrics we want to optimize for​. Compile takes several parameters that can be used to specify these values:

  - **optimizer**​ - the optimizer to use for our training step. This is an instance of the tf.train.Optimizer class.
  - **loss​** - a string or array of string representing the loss function that we want to minimize during our training step.
  - **metrics**​ - a list of metrics to be evaluated by the model during the training and testing phases.
  - **evaluate** ​- returns the values of loss and metrics of our model in test mode. This method accepts two parameters
    - **x**​ - the input data, which can be a tf.Tensor, an array of tf.Tensor instances.
    - **config**​ - an object containing optional fields that can be used for configuration
      - **batchSize**​ - ​the number of items to use in a batch . ​The value is 32.
      - **verbose** ​- should we run this method in verbosity mode. The
        default value is false.

- **predict** ​- generates prediction output for a list of inputs
  fit - trains the model for a given number of epochs. This method takes three parameters
  - **x**​ - a tf.Tensor instance or array of tf.Tensor instances representing our inputs
  - **y**​ - a tf.Tensor instance or array of tf.Tensor instances representing our outputs
  - **config**​ - an optional object that contains the following properties
    - batchSize​ -
    - epochs​ -
    - verbose​ -
- **summary**​ - prints a human readable summary of the layers of a model, which includes the names and types of all layers that comprise the model, the out shape of each layer, the number of weights contained in each layer, the input each layer receives (if applicable) and the total number of trainable and non-trainable parameters of the model. Figure 6.3 shows a sample output:

**Figure 6.3**

```javascript
/*
_________________________________________________________________
Layer (type)                 Output shape              Param #
=================================================================
dense_Dense3 (Dense)         [null,1]                  2
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________
*/
```

The summary method also takes several parameters that can be used to customize the output:

- **lineLength​** - a custom length, in number of characters, to use for a line.
- **positions** - ana ray of custom widths to use for each of the columns. These numbers can be fractions from 0 to 1 or absolute number of characters.
- **printFn** ​- a custom function that can be used to replace the default console statement that is printed by method.

Figure 6.4 demonstrates the output with each of these parameters provided:

**Figure 6.4**

```javascript
model.summary(30, [100, 50, 25], x => console.log("output: " + x));

/*
output: ______________________________
output: Layer (type)
output: ==============================
output: dense_Dense3 (Dense)
output: ==============================
output: Total params: 2
output: Trainable params: 2
output: Non-trainable params: 0
output: ______________________________
*/
```

Though there are more methods available to us on the `tf.Sequential` and `tf.Model` classes these are the ones that you will probably mostly interact with. To learn more about other methods provided by the Layers API it is recommended to read through the official documentation or source code.

## A Real-world Example: Recurrent Neural Network

TBD

## A Real-world Example: Autoencoder

TBD

## A Real-world Example: Convolutional Neural Networks

Convolutional neural networks (CNNs) are special type of artificial neural network that is used for image classification. The work by applying a series of filters to the raw pixels of an image to extract and learn specific features that is can use to perform the image classification task. This example comes directly from the TensorFlow.js documentation and has been changed to work in a Node.js environment so that we can train and save our model to the file system.

They are composed of the three following components:
Convolutional Layers - apply convolution filters filters to an image to perform feature extraction. These layers then usually pass their outputs to a pooling layer for further processing.

- Pooling Layers - these layers perform downsampling on our image data with the aim of reducing the dimensionality of the image to achieve a performance boost in processing time. These layers then pass their outputs to one or more dense/fully connected layers for processing.
- Dense/Fully Connected Layers - these layers perform the classification of the features extracted by the convolutional layers and downsampled by our pooling layers. These layers are called fully connected because each node is connected to every node in the previous layer.

## Creating Our First Layers

Now that we know the primary components of a CNN let’s review the steps that we will take to crate out CNN handwriting classifier.
First we will create a convolutional layer by using the tf.sequential constructor:

**Figure 6.x**

```javascript
const model = tf.sequential();
```

We then add our first layer, which will be a two-dimensional convolutional layer. Convolutions slide a special filter over an image to learn patterns about different parts of an image. For example, we may have one convolutional layer that has the sole job detecting eyes in an image. Figure 4.14 provides an example of how these filters work.

**Figure 6.x**
We will go deep into these structures in a later chapter, for now all we have to know is that TensorFlow.js makes it simple to add a two-dimensional convolutional layer by using the tf.layers.conv2d constructor.

**Figure 6.x**

```javascript
model.add(
  tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: "relu",
    kernelInitializer: "VarianceScaling"
  })
);
```

The snippet above adds a two-dimensional convolutional layer to our model. The tf.layers.conv2d constructor takes a configuration object that details the features of our layer.

The breakdown is as follows:

- inputShape - this determines the shape of the data that will be input into the layer. Since our input will be images of size 28x228 pixels we define a shape of [28, 28, 1], where the array represents the following values: [row, column, depth]. We chose a depth of 1 since our images are in black and white and are represented by one color channel.
- kernelSize - this controls the sliding of the colutional filter that we described above. By choosing a kernel size of 5 we specify our filter window to be 5x5 in size.
- filters - this value specifies how many filters of the size defined in our kernelSize property will be applied to the input. In our case we have selected 8.
- strides - we can think of the strides as the “step size” of our filter windows, where the step represents how many pixels we should slide our filters across each image. We specified 1 as our value so we will shift our filters across our images in 1 pixel steps.
- activation - this field represents the activation function that we wish to apply ot the date after each full convolution has completed. The Rectified Lienar Unit (ReLU) function is the usual suspect when it comes training machine learning models so we have decided to leverage it here.
- kernelInitializer - this property allows us to specify the method to use to randomly initialize our first set of model weights. In this case we chose to use variance scaling.

As mentioned above, convolutional layers are usually followed by pooling layers in convolutional neural networks. The pooling layers is used to downsample the model result that is returned from the convolution layer. The layer we will add is a max pooling layer, which will downsample our image by computing the maximum value for each of our sliding filters. The figure below illustrates what this process looks like.

**Figure 6.x**

Figure 4.16 demonstrates how to add a two-dimensional max pooling layer in one line of code using the TesforFlow.js layers API.

**Figure 6.x**

}));

The above snippet of code adds a two-dimensional max pooling layer to our CNN. The job of this layers it to take the result of our convolutional layer and downsample the result by computing the maximum value for each sliding filter. Like our 2D convolutional layer the `tf.layers.maxPooling2d` layer takes configuration object that describes the primary characteristics of the layer. The properties of this object are broken down as follows:

- poolSize - this determines the size of the pooling windows to be used on our input data. By specifying [2, 2] as our value we will apply pooling windows of
  model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2],
  strides: [2, 2][ 10 ]

size 2x2 pixels to our data.
strides - this field specifies the “step size” of the our pooling window as we slide
it across our image. In our example we chose a size of [2, 2], which means that the filter window will be slid over our image in steps of 2 pixels in the horizontal and vertical directions.
Our last step will be to add another set of convolutional and pooling layers to our neural network. We can easily do this as follows:

**Figure 6.x**

```javascript
model.add(
  tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: "relu",
    kernelInitializer: "VarianceScaling"
  })
);
model.add(
  tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  })
);
```

In the above snippet we add a new two-dimensional convolutional layer and specify it to have a kernel size of 5, 16 filters, a stride of size 1 and to use ReLU as an activation function and variable scaling for initializing our weights. We then pass the output from this layer to a two dimensional max pooling layer for downsampling. This max pool layer has a pool size of [2, 2] and will takes strides of [2, 2]. Our next steps will be to then flatten our output from this layer and feed it a fully connected layer, which we can create using the tf.layers.dense constructor.

**Figure 6.x**

```javascript
model.add(
  tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: "relu",
    kernelInitializer: "VarianceScaling"
  })
);
model.add(
  tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  })
);
```

In the example above we added two new layers to our CNN. Next we flatten this data so that we can classify the data. What is essentially doing is converting the output of our max pooling layer from a matrix into a one dimensional vector of features that can be fed to our dense layer. Don’t get too confused by the terms matrix in vector as we’ll cover them in the next chapter’s linear algebra review section. Next we will train our model and use it for inference on new data.

## Training Our CNN

Before we can use our convolutional neural network we will have to first train it. If you recall from the previous chapter this consists of decreasing the errors produced by our loss function. To start we’ll create a variable for our learning rate and use the TensorFlow.js standard library to select an optimization algorithm. For our CNN we’ll use Stochastic Gradient Descent, which we learned about in the last chapter. This is demonstrated in figure 4.20.

**Figure 6.x**

```javascript
const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
```

Next we’ll have to choose a loss function to minimize during our training step. A commonly used loss function for training convolutional neural networks is the cross- entropy loss function. This loss function measures the error between the probability distribution produced by the final layer of our model and probability distribution provided by our labels. What this means is that for images fed to model we can predict that it falls into the 0, 1, 2, 3, 4, 5, 6, ,7, 8 or 9 class with a certain level of probability. Values closer to 1 denote a higher probability that we have classified the digit correctly.

For our next step we will compile our model. Tensorflow.js provides a compile method on the model instance that we can pass a configuration object to as demonstrated below.

**Figure 4.20**

```javascript
model.compile({
  optimizer: optimizer,
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"]
});
```

In the snippet above we pass our Stochastic Gradient Descent optimizer instance and specify that we want to minimize categorical cross entropy as our loss function and that we want to increase the accuracy of our model.

> Note: Accuracy is one of several metrics used for evaluating classification models. It is the ratio of predictions our models predicted correctly to the total number of prediction that we made. For example, if our model made 100 prediction and got 90 of them right then our model has an accuracy of 91%. In actuality the measure is slightly more complicated but we will dive deeper into this measure in a later chapter. For now the explanation above will suffice.

Another measure we will have to take into consideration when training our model is the size of our data. To avoid processing too much data at once we split our data set into smaller batches to be processed separately. Remember that we are training our model in a browser environment, where memory is limited. In our CNN we’ll process our data in batches of 64. We’ll also train our data on 100 batches and test our accuracy on 1,000 items every five batches. Let’s create some constant variables to store these values.

**Figure 6.x**

```javascript
const BATCH_SIZE = 64;
const TRAIN_BATCHES = 100;
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;
```

Next we will write our training loop for our specified number of training batches. Within the loop we take a batch of from our data set, as specie by our batch size. If our iteration is a multiple of (every 5 batches) we will reshape the data and store it for validation later on in the process. Next we call the fit method on our model instance to actually train our model and update our parameters. This method takes the following parameters:

- features - a vector of features. Since our feature is currently a matrix we must reshape it into a vector to be used with this method
- labels - a vector of our labels.
  A configuration object consisting of the following values:
- batchSize - the amount of images that we want to include each training batch
  validationData - the data that we will use the measure the accuracy of our model
- epochs - the number of times to process each batch. We specify 1 since we only want to train each batch once.
  The fit method returns an object that contains logs of metrics. We store this object in variable and use it to log our loss and accuracy at the end of each iteration. This entire process is detailed in the code below.

**Figure 6.x**

```javascript
// App.js
for (let i = 0; i < TRAIN_BATCHES; i++) {
  const batch = data.nextTrainBatch(BATCH_SIZE);
  let testBatch;
  let validationData;
  if (i % TEST_ITERATION_FREQUENCY === 0) {
    testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
    validationData = [
      testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]),
      testBatch.labels
    ];
  }
  const history = await model.fit(
    batch.xs.reshape([BATCH_SIZE, 28, 28, 1]),
    batch.labels,
    {
      batchSize: BATCH_SIZE,
      validationData,
      epochs: 1
    }
  );
  const loss = history.history.loss[0];
  const accuracy = history.history.acc[0];
  console.log("LOSS: ", loss, "ACCURACY: ", accuracy);
  printResults();
}
```

After running our model you’ll see a plot of the results printed to the screen. If we examine the output you’ll see that our loss has been mitigated over the number of batches we run while the accuracy increases. If we examine the the results you should see that it correctly classified the majority of the digits in our test data set. In a later chapter we will go deeper into the mechanics of how convolutional neural networks work.

**Figure 6.x**

```JavaScript
/*
Model run with the following arguments:
---------------------------------------
Epochs: 20
Batch Size: 128
Model Save Path: ./trained_models

TensorFlow.js deprecation warnings have been disabled.
_________________________________________________________________
Layer (type)                 Output shape              Param #
=================================================================
conv2d_Conv2D1 (Conv2D)      [null,26,26,32]           320
_________________________________________________________________
conv2d_Conv2D2 (Conv2D)      [null,24,24,32]           9248
_________________________________________________________________
max_pooling2d_MaxPooling2D1  [null,12,12,32]           0
_________________________________________________________________
conv2d_Conv2D3 (Conv2D)      [null,10,10,64]           18496
_________________________________________________________________
conv2d_Conv2D4 (Conv2D)      [null,8,8,64]             36928
_________________________________________________________________
max_pooling2d_MaxPooling2D2  [null,4,4,64]             0
_________________________________________________________________
flatten_Flatten1 (Flatten)   [null,1024]               0
_________________________________________________________________
dropout_Dropout1 (Dropout)   [null,1024]               0
_________________________________________________________________
dense_Dense1 (Dense)         [null,512]                524800
_________________________________________________________________
dropout_Dropout2 (Dropout)   [null,512]                0
_________________________________________________________________
dense_Dense2 (Dense)         [null,10]                 5130
=================================================================
Total params: 594922
Trainable params: 594922
Non-trainable params: 0
_________________________________________________________________
Epoch 1 / 20
eta=0.0 ==============================================================================================================================>

. . .

==============================================================================================================================>
197009ms 3863us/step - acc=0.997 loss=0.00950 val_acc=0.994 val_loss=0.0321
Epoch 20 / 20
eta=0.0 ==============================================================================================================================>
197571ms 3874us/step - acc=0.997 loss=0.00864 val_acc=0.994 val_loss=0.0291
*/
```

## Summary

In this chapter we took a look at the TensorFlow.js Layers API, which is inspired by the popular Keras library that currently ships with the Python version of TensorFlow. We learned some of the most useful methods provided by the layers API and saw how it makes creating deep neural networks slightly easier than working with the core API directly. As a real-world example we trained a Convolutional Neural Network (CNN) and taught it how to recognize handwritten digits. This example comes directly from the TensorFlow.js documentation. So far we have been running our examples using Node; however, in the next chapter we'll learn how to import an existing model and interact with it in the browser by importing the model that we trained in this chapter.
