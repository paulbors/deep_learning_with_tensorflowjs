# Introduction to Artificial Neural Networks

In the last chapter we setup our Node.js and TensorFlow.js development environment and ran a “hello world” of sorts. In this chapter we will delve deeper into machine learning theory in order to develop the intuition required to build deep learning models with TensorFlow.js.
To accomplish this we will cover the following topics:

- Understand how deep learning relates to machine learning and artificial intelligence
- Cover the differences between supervised and unsupervised learning
- Review artificial neural networks, their components and how they are used to perform inference
- Discuss how to train an ANN
- Look at different activation functions, cost functions and loss minimization algorithms we can use to train our models
- Understand the problems of overfitting and underfitting as well as how to address them

## Machine Learning

Machine Learning is the practice of using algorithms to gain insight from a data set, learn from this data set and then make some kind of prediction from the data. The key point here is that the program is able to learn from the data that it is given. Rather than writing a program that attempts to create the impression of being intelligent by anticipating and a series of edge cases and coding actions based on the user interaction.

Take chatbots for example. In the early days developers building a chatbot to order pizza would create a program that may look like the pseudo code shown as follows:

**Figure 3.1**

```
// psuedo code for a pizza delivery chatbot

print(“Hello, thanks for calling PizzaBot! How may I help you?”)

user_input = “some user response”

if user_input == “I would like to place an order” or “I would like a pizza for delivery”:
    print(‘Sure, I can help with you that. Press 1 for pizza, 2 for other menu items or 3 for condiments. Please note that you can talk to a real human being by pressing 0 at any time’)

user_input = 1 // this comes from the user

if user_input == 1:
    print(“The available pizza sizes are small, medium and large. Please say 1 for small, 2 for medium and 3 for large”)

user_input = “some response from the user”

if user_input:
    print(“Would you like toppings on your pizza? Select 1 for yes or 2 for
no.”)

if user_input === 1:
    // continue with the toppings selection flow
if user_input == 2:
    // try to upsell other items
if user_input == 2:
    // kick off the other menu items flow
if user_input == 3:
    // kick off the condiments flow
if user_input == 0:
    // forward user to a real human
else:
    print(‘Sorry, I did not understand your input’)
```

The example above starts off as what appears to be a simple block of code. After all, all we are doing is inviting the user to create an order while we attempt to take the user through different order flows based on their responses to certain questions. However, as we get into actually ordering the meal, which could consist of multiple pies, other menu items and different kinds of drinks the pizza could have been prepared and ready for delivery. By the time the user completes the flow, assuming there were no misunderstanding between your intentions and the computer program you could have your pizza order completed in about ten minutes tops. Maybe our example is a little too basic and it would probably be smarter to check for certain keywords rather than matching on entire sentences but you get the idea that trying to write a program that handles all of the edge cases that can occur in human speech would be a pretty lofty task for any skilled programmer.

In the field of machine learning, we do not try to anticipate edge cases since there can be many different ways to say "I would like to order a pizza." Human beings are used to having the experience of naturally expressing ourselves by talking to a human rather than to a machine. A modern approach, leveraging machine learning, would train a model to understand a user’s intent by learning to classify them from a data set containing the many different ways that a user can complete the flow. For example, the model would understand the sentence "I'd like two pepperoni pies" and "Give me two pies with pepperoni" as having the same intent and can decide to take the user to next part of the flow without defaulting to a message similar to "Sorry, I did not understand your request" if the user does not respond with an answer that matches a case the developer programmed for.

Machine Learning usually consists of two major types of learning :

- Supervised learning
- Unsupervised learning

> Note: In this book we will use the terms algorithm, model, neural net, net or ANN but they all refer to the same thing.

## Supervised Learning

A model that learns through the process of supervised learning learns and makes inferences (inferences are another name for predictions) on data that has been pre-labeled.
You can think of this like receiving the answers for an example and studying the questions. When you actually sit for the exam you get new but similar questions to the ones you studied for and you can use what you’ve learned from your studies to correctly answer the questions on the example. The more you study the more questions you'll see and the higher the probability of you getting the correct answer for new questions you haven't seen before increases. This a lot like how supervised learning works in machine learning.

Usually the features of a data set that is fed are labelled and the model learns to categorize items base on these labels. As the model receives more and more labelled data it starts to recognize patterns that it can use to derive characteristics about the data in the data set. After the model has seen enough data it can take its knowledge of the labelled data and make an inference on what the new data is by comparing it with what it knows about the data set that it learned from. In this type of learning the model is shown the correct answers and learns how to classify or predict a value based on new unseen data.

A good example of this is the photo tagging feature that is popular with many social media applications. The way that your favorite photo sharing app is able to recognize photos of you, your friends and family works very similar to the process described above. When you upload photos to the cloud you are more than likely providing some machine learning model a data set, which in this case is your photos. If you elect to tag yourself and your friends in these photos you are providing labels for the model work with. Once the model has seen enough photos of you, your aunt May and uncle Ben it will have enough data to automatically recognize a new picture of you or a family member and tag it. We will be building a simplified version of a similar model in a later chapter of this book.

## Unsupervised Learning

If a model is able to learn from the data and create its own labels we refer to this process as unsupervised learning. If we take our example of the social photo sharing application from above we can take the initial upload of photos.

Imagine that you uploaded 2,000 photos to this social media site. Though the model doesn’t know who you are or who the other people in your photos are, it can start to recognize faces and apply its own labels to the data before you even begin to label them by tagging them. Behind the scenes the model will create categories for people that look like you and people that look like Uncle Ben, people that look like Aunt May and so and so on.
This process of automatically making inferences from unlabelled data is called unsupervised learning (i.e. it doesn’t have the answers upfront and learns the features and classifies them without assistance).

# Artificial Neural Networks

Artificial Neural Networks (ANN) are a computing systems inspired by the human brain. The primary components of an ANN are neurons, connections, input layers, hidden layers and output layers. These are briefly explained below:
Input layer: Layer that contains our initial data that will flow through to other layers of neurons of processing (i.e. a list of a user's tagging of favorite films and watched films)

- **Hidden layer**: A type of layer that sits between input and out players and processes data. This processing comes in the form of weighted input that are passed through an activation function.
- **Output layer**: The final layer of neurons that produces outputs for our neural network. This represents the result (i.e. given a user's past favorites and watched movies send.

## Types of Neural Networks

This section will explain two important types of neural networks, which are the simple perceptron and multi-layered artificial neural networks.

### Perceptron

The simplest ANN is the Perceptron, which is a linear classifier that is used for binary predictions (i.e. a predictions that fit into one of two categories). This simple ANN consists of an input layer consisting of our training data, a single hidden layer that takes our inputs, applies a weighted sum and passes them through an activation function to produce an output. We will be building a simple perceptron in the next chapter. An example of how this ANN looks is as follows:

**Figure 3.1**

![](./images/3.1.png)

### Multi-layered ANN

When data flows through a hidden layer its output is passed to the next layer for additional processing until it gets to the output layer. Each layer may have a particular function to perform. For example, an ANN that attempts to recognize images will contain a hidden layer with the sole purpose to recognize edges, while another layer's job is to detect eyes, another layer's job may be to detect lips and so and so on. The more layers added to an ANN the better it may or may not perform. ANNs containing more than one hidden layer are referred to as deep neural networks. This is also where the term deep learning comes from. An example multi-layered ANN appears as follows:

Figure 3.2

![](./images/3.2.png)

We’ll see that TensorFlow.js, particularly the Layers API, provides different kinds of layers for us to use out of the box when building our models.

## Activation Functions

The activation function of a given neuron defines the output of that neuron. This is based on how different neurons in the brain fire and become activated based on the sensation it receives.

For example, if you hear a song you like certain neurons in your brain will light up and respond to the stimuli. If you in turn hear a song you can’t stand another set of neurons will light up to tell your brain that you don’t like what you’re hearing.

This is similar to how activation functions will light-up based on certain input values. For example, if the model is looking for a photo of you rather than aunt May the neurons that have been trained to recognize the features that identify who you are will return a stronger signal for images of you rather than those of other people. In our case the brighter light is a number closer to one. We’ll dive deeper into this as we take a deeper look at some activation functions below.

### Sigmoid Activation Function

The sigmoid activation function normalizes the input to some value that fits between 0 and 1. When plotted the graph of the line created takes the form of an S-shape that we call an sigmoid curve. The formula and plot are represented as follows:

**Figure 3.3 - Sigmoid Formula**

![](./images/3.3.png)

**Figure 3.4 - Sigmoid Formula Plot**

![](./images/3.4.png)

### ReLU Activation Function

ReLU is an activation function that returns 0 if our input is a negative value or its actual value if it is positive. The formula and plot appear as follows:

**Figure 3.5**

![](./images/3.5.png)

**Figure 3.6**

![](./images/3.6.png)

### Softmax Activation Function

The Softmax activation function is used for multi-class classification. This function helps us calculate the probability distribution of a given event over n different events. These probabilities help us determine the class for inputs that are provided to our model. The formula appears below for further examination:

**Figure 3.7**

![](./images/3.7.png)

### Tanh Activation Function

An activation function similar to the sigmoid function is the Tanh function, or hyperbolic tangent. The tan-h function also follows an S-shape but the values range from -1 to 1. The formula and plot appear as follows:

**Figure 3.8**

![](./images/3.8.png)

There are other activation functions but these four are some of the most commonly used ones. What activation function we use for our layers depends on what problem we are trying to solve. If the problem is a binary classification problem we may want to use activations functions that perform well at this particular task. If we want to solve a multi class classification problem it makes more sense to use an activation function that performs well on solving these types of problems.

## Model Training

When we are training a machine learning model we are essentially solving an optimization problem where we are optimizing for the weights in each of our neurons. To achieve this we have to minimize a given loss/cost function, which we covered in the previous section. To minimize the costs in our models we usually chose an optimization algorithm (called optimizer in TensorFlow.js). The following sections provide a high level overview of the steps that are taken to train a model.

### Forward Propagation

Forward propagation is the process of providing our artificial neural network with input data that are applied weights, processed in a layer by an activation and output to the next layer for further processing. As explained in the previous sections, when our data reaches the final, or output, layer we receive a prediction from our model. The strength and accuracy of our predictions depends on the set of weights that have been selected for our artificial neural network. As we continue feed our data forward through the model we may want to tweak our weights until we have an optimal set of values for our weights such that our predictions are strong. This process of tweaking our weights is called “training” our model.

### Cost Functions

In the previous section we mentioned that when training a model our goal is to try to find the optimal set of weights to apply to our data as it flows through the layers of our ANN. The better the sets of weights we use in our model the better the model will perform. A measure of the performance of our models is a cost function. The larger the cost the lower the quality of the predictions of our model. This means that the cost function is a measure of how wrong our model is when it attempts to estimate the relationship between inputs and outputs. Mathematically this is represented as the difference between our predicted values and the actual value.

When we train our model we are effectively minimizing the measure of the cost (also called loss) of our cost function. This is in essence the role of our model—to minimize the cost function until we have a model that can accurately predict values when given new input. This process of minimizing the cost function is called training the model and when the cost function decreases we say that the model is learning. We'll look at different techniques that we can use to minimize our costs but for now we look at a popular cost function, the mean square error.

### Mean Square Error

The mean square error of a model is the average of the squares of the errors in a model (i.e. the squared differences between our actual and predicted values. This measure is always positive and and provides us with information of the quality of our model. Figure 3.9 displays the formula:

**Figure 3.9**

![](./images/3.9.png)

### Gradient Descent

Previously we mentioned that models learn by minimizing a cost function. This process can be computationally expensive; however, there exists several algorithms that help achieve this goal in a more efficient manner. Gradient descent is an optimization algorithm that attempts to find the local or global minimum of a function.
The first objective of gradient descent is to discern the direction the model move toward in order to reduce errors in our model. As explained above, these errors are the differences between our actual values and our predicted values. This equates to adjusting our model weights in a way such that the cost is minimized and the predictions become stronger.

Each time we pass through the model we adjust the weights so that the cost continues to decrease and our predictions begin to become stronger. As we continue to move along our cost function we keep moving in the direction that minimizes the cost function until its output is zero. When we reached the point where the cost is zero we have optimized our model weights so that we minimize the cost function. This is much faster than randomly trying to adjust each weight until they were optimized. By performing gradient descent we are able to find the optimal weights faster.

The figure below demonstrates how gradient descent looks when graphed.

**Figure 3.10**

![](./images/3.10.png)

The explanation above may seem like magic but in actuality, gradient descent is used in another technique used to train models called backpropagation.

### Backpropagation

Backpropagation is a technique applied to artificial neural networks to calculate the gradient required to find the optimal weights used in the network. The way that this is achieved is by minimizing a cost function by taking its derivative until we find the point where its value is the lowest, which is zero. If you think back to any courses on Calculus this is solving an optimization for a global maximum or minimum of a function. In our case we are looking to find the global minimum of our cost function. Once we find this value we can be certain that we have found the values of our weights that most closely approximates the function that our artificial neural network is attempting to solve for.

The way that backpropagation works is fairly straightforward. First we select a random point on the plot of our function and take a step forward and a step backward and evaluate the slope of the line at these points. If the slope of our cost function increases in value then we know we are moving in the wrong direction. If the slope decreases from its initial value we know that we are moving in the correct direction. Our next step in the process is to continue taking steps in the direction that our slope decreases until we arrive at a global minima, where the slope is equal to zero. An illustration of this process can be found in figure 3.11 below.

**Figure 3.11** (TODO: annotate this to show steps)

![](./images/3.11.png)

### Learning Rate and Epochs

It is important to note that there are two features that impact how quickly we find this global minima, if we ever find it at all. These two features are the learning rate epoch where the learning rate determines how large of a step we take as we move along the cost function’s curve while the epoch is the number of times we will attempt to pass through the neural network adjusting our position along the curve during each iteration. An important relationship between the learning rate, also called alpha, is that the smaller the learning rate the longer it will take to find a global minima. We can speed up this process by selecting a larger learning rate; however, we may run the risk of missing our global minima if this value is too high.

An important relationship also exists between the number of epochs we choose in our learning process, which consists of a full forward pass (forward propagation) and backwards pass (backpropagation). The more times we pass through our network, also referred to as training steps. The number of epochs and the amount of data that is passed through the network can determine the effectiveness of the learning process. Remember that with each pass we adjust the weights of our artificial neural network so it should make sense how it’s highly doubtful that we can get the correct combination of weights on a single pass.

By determining the number epochs we decide how many times we would like to go through the process and continue along with the process explained above. If we go through enough iterations then we have a better chance of finding the global minima for our cost function and the optimal set of weights of our ANN. However, just like selecting a learning rate, selecting the number of epoch comes with a tradeoff. The more epochs selected and the larger the amount of data that has to pass through the artificial neural network the longer it will take to train our model and more resources may be required to train it.

### A Note on Overfitting

Overfitting occurs when the models does really well at classifying data in the training set but does poorly on new data that it has never been seen before. To visualize this take a look at figure 3.10, which shows what an overfitted model potentially looks like. We will not dive to deeply into this for the purposes of this book; however, we will include various techniques to identify and prevent overfitting.
One way to eliminate overfitting is to use a larger data set. The more data that is fed to a model the better it will perform in the training process. Another technique is to use data augmentation, in which we slightly change the data before training our data. For example, we can augment a set of images we plan to use for a facial recognition program by cropping some of images, rotating them or converting them from color to grayscale. We can also reduce the complexity of our model by dropping the number of layers for example, of our model to reduce overfitting. If we use a function that has too many degrees it can overfit our data set. Another technique that is used to reduce overfitting is dropout. What this does is randomly ignores (drops out) nodes out of the network to help the model better generalize about new data that it has never seen before.

**Figure 3.12**

![](./images/3.12.png)

### A Note on Underfitting

Underfitting occurs when a model does poorly performing inference on the data it was trained on. This happens when the model metrics are poor. For example, the cost is high or the accuracy is very low. Ways to improve model performance are to increase the complexity by changing the type of layer we’re using, the number of neurons in each layer or the number of layers themselves. Increasing the size of the data set will also help improve model performance. In addition, we can add additional features. Another technique is to reduce the amount of dropout in the model. Figure 3.11 shows an example of what underfitting looks like graphically.

**Figure 3.13**

![](./images/3.13.png)

## Summary

In this chapter we were able to gain a very high-level overview of what deep learning is and how it fits into the grand schema of the fields of Machine Learning and Artificial Intelligence. We took a dive into Artificial Neural Networks and their major components including the input, hidden and output layers. We then learned how ANNs are trained by minimizing a cost function. We learned the roles of activation functions and the process of backpropagation and gradient descent to minimize these costs. We also learned how the learning rate can impact our results. Last we identified the different types of Artificial Neural Networks.

In the next chapter we will take everything that we learned in this chapter and will use TensforFlow.js to build our first Artificial Neural Network.
