# Chapter 5: Understanding the TensorFlow.js Core API

In the last chapter we build our very first Artificial Neural Network (ANN) using the TensorFlow.js layers API. In this chapter we will take a dive into the TensorFlow.js Core API, which the Layers API that we received a brief introduction to is build on top of. Our aim is to explore this low-level API to learn how it is structured as well as what methods and data structures it provides to allow us to create deep learning models that can be run in the browser. By the end of this chapter we should:

- Understand the role of Tensors
- Be able to differentiate between scalar, vector and matrix Tensors
- How to perform operations on Tensors
- Build an Artificial Neural Network from scratching using the Core API
- How to train an ANN and predict results
- How to test the model on unseen data

> Note: To keep the focus on the API an its methods we will be using the `tfjs-node` project. It's important to note that at the time of this writing this API is experimental and is not ready for production use. However, since TensorFlow is coming to Node.js it is worth gaining on understanding of how TensorFlow can be used in a Node environment. Since this chapter focuses on the TFJS API methods, which are the same between the `@tensorflow/tfjs` and `@tfjs-node` packages the same examples that are presented here should also work in the browser. Further instructions on how to run these code examples can be found in the code samples directory.

## Tensors

Tensors are the fundamental building block of TensorFlow.js. They are the essential data structure and are typically used to represent our data inputs, outputs and transformations between layers in our network. Tensors usually take on different shapes depending on our data and can represent scalar values, one-dimensional, two-dimensional, three-dimensional and four-dimensional values. In the next section we’ll briefly explain the difference between these shapes. For now we will review how to create tensors in our applications.

Tensors are created by the ​`tf.tensor​` factory function, which returns a `tf.Tensor​` object. All `tf.Tensor` objects are immutable multidimensional array of numbers that has a specific shape and a type. Figure 5.1 demonstrates how we can create a one-dimensional tensor by passing an array of numbers to the `tf.tensor` factory function.

**Figure 5.1**

```javascript
const vector = tf.tensor([1, 2, 3, 4, 5]);
vector.print();
```

In the code snippet above we created a simple vector tensor that contains 5 elements and has one dimension. The reason that this is a vector tensor is because since the data has only one dimension and five elements (1x5). You can

think of a dimension as a column in a matrix. In the next example we will create a two-dimensional matrix and for the purpose of this tutorial you can think of a matrix of a vector of multiple columns.

**Figure 5.2**

```javascript
const matrix = tf.tensor([[0, 0, 0], [1, 1, 1]]);
matrix.print(); // => Tensor[[0, 0, 0], [1, 1, 1]]
```

The example above creates a 2x3 matrix because we have two columns (or two elements) in our top level array and each position in this array contains another array that contains three elements each. By inspecting the ​shape property on our matrix output we’ll see that it returns an array representing the shape of the matrix, ​[2, 3]​.

Another way to create a tensor is to pass the `tf.tensor` factor a flat array of values and second argument that contains a second flat array that specifies the shape. So if we wanted to create the same 2x3 matrix that we created in our previous example we can do the following to achieve the same result:

**Figure 5.3**

```javascript
const matrix2 = tf.tensor([0, 0, 0, 1, 1, 1], [2, 3]);
matrix2.print(); // => Tensor [[0, 0, 0], [1, 1, 1]]
```

By default, the numbers that are used to create our vectors are 32-bit floats. However, what happens when we want to use 32-bit integers instead? Luckily for us the solution is simple since we can easily specify the datatype of our values by passing a third argument to the `t​f.tensor​` utility function.

**Figure 5.4**

```javascript
const int32Matrix = tf.tensor([0, 0, 0, 1, 1, 1], [2, 3], "int32");
int32Matrix.print(); // => Tensor[[0, 0, 0],[1, 1, 1]]
int32Matrix.dtype; // => ‘int32’
```

In the examples above we used the print function to print the result of of our calls to `tf.tensor`​. This is actually a convenience function that “pretty prints” what we would expect our tensor to look like in the world of mathematics. However, if we inspect our tensor directly we’ll realize what we get back from `tf.tensor` is a POJO, or plain old JavaScript object. If take a look at the properties of the `tf.Tensor`​ object we’ll get a better understanding of the information imbedded within the structure:

- **dataId**​ - the ID of the bucket that contains the data for a particular tensor. This important because multiple arrays can point to the same bucket.
- **dType**​ - specifies the data type of the tensor. All elements in a tensor must be o the same data type and can be one of the following types: float32, int32, bool, and complex64. If you look through the TensorFlow.js source code you will see that types map to the Float32Array, Int32Array, Uint8Array and Float32Array types respectively. Though elements in a tensor must be of the same type it is important to note that they can be cast to different types using a utility function that we will look at later in this chapter.
- **id**​ - represents the unique identifier used to identify a specific tensor instance
- **isDisposed​** - specifies whether or not a tensor has been disposed from memory
- **isDisposedInternal**​ - specifies whether or not a tensor has been disposed from memory
- **rank**​ - specifies the number of dimensions of the tf.Tensor object. The terms order, degree and n-dimension are also used. Each TensorFlow.js rank corresponds to a particular math entity, as demonstrated in the table below:

| Rank | Entity | 
| :---:| :---: |
| 0 | Scalar |
| 1 | Vector | 
| 2 | 2-Matrix |
| 3 | 3-Matrix |
| 4 | 4-Matrix |
| ... | ... |
| 5 | n-Tensor |

- **rankType**​ - represents the rank type of an array, which can be any of the values ‘R0’, ‘R1’, ‘R2’, ‘R3’, ‘R4’, ‘R5’ or ‘R6’.
- **shape** -​ represents the shape of the tensor, which is the number of elements in each dimension. Another way to view this is as how many columns and rows are contained within it. For example, a matrix consisting of two columns and five elements in each column would have a shape of 2x5.
- **size**​ - specifies the number of elements in a tensor. This is typically the result of multiplying the number of columns by the number of elements in each column. So for for example, a 3x3 matrix would have a size of 9.
- **strides**​ - represents the number of elements to skip in each dimension when indexing.

_A Note on Mathematical Explanation of the terms Scalar, Vector, one-dimensional and N-dimensional. (optional section)_

In this chapter we will use a lot of terms that come from mathematics, more specifically linear algebra. Some readers may have little linear algebra experience or it may have been years since they completed a college course on the subject so we will briefly define these terms below. This section is optional but may be useful since these terms will come up again later in this chapter and throughout the rest of the book.

- **Scalar** - an element of a field that is used to describe a vector space. For purpose we can just think of a scalar as a number like 1, 10, 100 or -0.9. They usually describe the magnitude of something such as the temperature, cost of a car or likelihood of a person having diabetes.
- **Vector** -  For our purposes we can think of a vector as a one-dimensional array containing 1 or more values or simply just a list of numbers. For our purposes vectors will usually represent an input into or an output from a neural network.
- **Matrix** - A matrix is sort of like a vector but is in a higher dimension. We can think of a matrix as a multi-dimensional array. We will typically use them as data that lives in layers of our artificial neural networks.
- **Rank** - In linear algebra the textbook definitions states that rank describes the dimension of the vector space generated by its columns. For our purposes we can think of the rank being how many rows "high" or columns "wide" a matrix. For example, a 2x5 matrix has a row rank of 2 and column rank of 5.
- **Shape** - The shape of matrix is simply the number of rows and columns it contains. For example, the following matrix has a size of 4x4 because it has four rows and columns:

```javascript
     [[1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1]]
```
- **Size** - The size of a matrix relates to the shape in that it defines how many items are contained with the matrix. In our 4x4 matrix above we can say it has a shape of 16 items because if were to count all of the ones we would see that we have 16. However, we know form elementary math that we can simply multiple the rows by the columns to find out how may items a matrix contains (i.e. 4 * 4 = 16 elements in our matrix).

You may be wondering why TensorFlow uses vectors to perform machine learning tasks it primarily because the operations that can be performed on these structures allows us to train models with a high degree of efficiency. Later in this chapter we’ll see how matrix multiplication can help us perform complex calculations in very few steps. Since just about any real-world data can be represented as a vector this is good news for us as machine learning engineers. If you would like to take a deeper dive into Linear Algebra I would recommend reviewing a textbook on the subject to gain a deeper understanding since going deeper than these simple explanations is definitely out of the scope of this book.

### Tensor Types

This section will explain the different Tensor types that are provided by the TensorFlow API. Though some of these concepts are derived from the subject of Linear Algebra we will keep the math involved to a minimum and will focus primarily on the code.

The different types of tensors that we will typically see in our machine learning models are as follows:

- **Scalar**​ - a tensor of rank 0. These values can be thought of as single numbers that are not contained in an array, like the number 33 for example.
- **Vector​** - a tensor of rank 1. This can be thought of as a tensor that contains an array of one column. We also refer to this as a one-dimensional or 1D tensor.
- **2D Matrix** ​- a tensor of rank 2. This can be thought of as a tensor that contains an containing two arrays of values. We also refer to this as a two-dimensional or 2D tensor.
- **3D Matrix**​ - a tensor of rank 3. This can be thought of as a tensor that contains an containing three arrays of values. We also refer to this as a two-dimensional or 3D tensor.
- **4D Matrix**​ - a tensor of rank 4. This can be thought of as a tensor that contains an containing four arrays of values. We also refer to this as a two-dimensional or 4D tensor.

### Utility Methods for Creating Tensor Types

In the previous section we took a look at some of the different types of tensors provided to us by TensorFlow.js. We also took a look at how we can create tensors of different shapes using the `t​f.tensor​` utility function. However, TensorFlow.js provides additional utility functions that make it convenient for us to create different types of tensors in one line of code.

#### Creating Scalar Tensors

To easily create a scalar tensor, which is a tensor of rank 0, we can use the `tf.scalar` utility method. It similar to `tf.tensor` in that it returns a `tf.Tensor` (actually a `tf.Scalar`) object; however, the shape is implied by the function name and all we have to do to create a scalar value is pass the value and data type to this function.

Figure 5.5

```javascript
const scalar = tf.scalar(3.14);
scalar.print(); 

Tensor
    3.140000104904175
```

#### Creating Tensors of Different Dimensions

TensorFlow.js provides helper methods that will allow us to create matrices of up to six dimensions. They are `tf.tensor4d`, `tf.tensor5d` and `tf.tensor6d`. They work similar in fashion to the three methods we previously looked at. Feel free to experiment with them.

#### Creating 1D Tensors

To create a one-dimensional (1D) tensor the TensorFlow.js library provides us with another utility method called `tf.tensor1d`. Though we can create a tensor of one dimension using `tf.tensor()` it is recommended to use `tf.tensor1d()` to be explicit and make our code more readable.

```JavaScript
const matrix1d = tf.tensor1d([3, 2, 1]);
matrix1d.print();

Tensor
    [3, 2, 1]
```

#### Creating 2D Tensors

Similar to how we can create a one-dimensional tensor using `tf.tensor1d` we can create a two-dimensional tensor using the `tf.tensor2d` method.

**Figure 5.7**

```JavaScript
const matrix2d = tf.tensor2d([5, 4, 3, 2, 1, 0], [3, 2]);
matrix2d.print();

Tensor
    [[5, 4],
     [3, 2],
     [1, 0]]
```

#### Creating 3D Tensors

If we want to create a tensor that is of three dimensions we can make use of the `tf.tensor3d` method provided by the TFJS library.

**Figure 5.8**

```JavaScript
const matrix3d = tf.tensor3d([[[1], [2]], [[3], [4]]]).print();
matrix2d.print();

Tensor
    [[[1],
      [2]],
     [[3],
      [4]]]
```

#### tf.reshape

When working with tensors we may need to reshape, to a 2x3 matrix from a 3x2 matrix for example. In order to achieve this we can leverage the `reshape` method that exists on tensor instances. One trick to keep in mind is that we can reshape a tensor to any dimension as long as the product remains the same. Figure 5.x demonstrates how we can reshape using the `reshape` method that exists on tensors.


```JavaScript
// create a two-dimensional tensor
const matrix2d = tf.tensor2d([5, 4, 3, 2, 1, 0], [3, 2]);
matrix2d.print();

Tensor
    [[5, 4],
     [3, 2],
     [1, 0]]

matrix2d.reshape([2, 3]).print();

Tensor
    [[5, 4, 3],
     [2, 1, 0]]

matrix2d.reshape([1, 6]).print();

Tensor
     [[5, 4, 3, 2, 1, 0],]


matrix2d.reshape([6, 1]).print();

Tensor
    [[5],
     [4],
     [3],
     [2],
     [1],
     [0]]
```


### Performing Operations on Tensors

When building our deep learning models we often have to perform operations on our tensors. These operations are usually mathematical in nature and involve adding, subtracting, multiplying and dividing tensors. However, methods are provided to allow us to easily create Tensors pre-filled with specific values. We will explore these methods in these sections.


#### tf.abs

In order to compute the absolute value of all values in a tensor we can use the `tf.abs` method. a simple example of how to use it appears below.

```JavaScript
const absTensor = tf.tensor1d([-10, -20, -30, -40, -50]).abs();
absTensor.print();

Tensor
    [10, 20, 30, 40, 50]
```

#### tf.ceil

To compute the ceiling of a given tensor we can use the `tf.ceil` method. Notice that all numbers will be rounded up even though they are closer to their lower bound.

```JavaScript
const ceiled = tf.tensor1d([.6, 1.1, -3.3]).ceil();
ceiled.print();

Tensor
    [1, 2, -3]
```

#### tf.floor

TFJS provides a method that is similar to `tf.ceil` called `tf.floor` method that will floor the values in the Tensor that we create. This is the exact opposite behavior of `tf.ceil`. In the example below notice how all numbers that were specified in our previous example will be rounded down even though they are closer to their higher bound.

```JavaScript
const floored = tf.tensor1d([.6, 1.1, -3.3]).floor();
floored.print();

Tensor
    [0, 1, -4]
```

#### tf.range

At times we will want to create a 1D tensor that spans over a range. We can easily accomplish this using the `tf.range` method. Below we see an example of how to accomplish this. 

```JavaScript
const range = tf.range(0, 100, 10);
range.print();

Tensor
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
```

#### tf.linspace

The `tf.linspace` method is similar to `tf.range` however, it provides an evenly spaced sequence of numbers for the interval provided to it, where the last argument is the number of values to generate.

```JavaScript
const spaced = tf.linspace(0, 1, 10)
spaced.print();

Tensor
    [0, 0.1111111, 0.2222222, 0.3333333, 0.4444444, 0.5555556, 0.6666667, 0.7777778, 0.8888889, 1]
```


#### tf.fill

If we want to create a Tensor pre-filled with some scalar value TFJS provides a method called `tf.fill` that will do just that for us. Creating a 2x3 Tensor pre-filled with the number 5 is demonstrated in the following snippet.

```JavaScript
const filled = tf.fill([2, 3], 5);
filled.print();

Tensor
    [[5, 5, 5],
     [5, 5, 5]]
```

#### tf.zeros

Pre-filling Tensors with scalar values is so popular that TFJS provides methods that allow us to pre-fill a Tensor with zeros. Running the commands below will create a 5x5 matrix pre-filled with zeros.

```JavaScript
const zeros = tf.zeros([5, 5]);
zeros.print();

Tensor
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]
```

#### tf.ones 

Pre-filling a Tensor with ones is also a popular feat to the point that TFJS provides a `tf.ones` method that will create a Tensor of a specified dimension for us. The example below demonstrates how we can create a 4x4 Tensor full of ones.

```JavaScript
const ones = tf.ones([4, 4]);
ones.print();

Tensor
    [[1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [1, 1, 1, 1]]
```


#### tf.eye

The `tf.eye` method will create a Tensor that contains a special type of matrix called an identity matrix, where the number passed as a parameter specifies the number of rows and columns of the matrix. Two simple examples appear below.


```JavaScript
// create a 5x5 identity matrix
const eye5 = tf.eye(5);
eye5.print();

Tensor
    [[1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1]]

// create a 10x10 identify matrix
const eye10 = tf.eye(10);
eye10.print();

Tensor
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
```


### Building Simple ANNs Using the Core API

In this section we will use the core API to build several very simple Artificial Neural Networks (ANN).

#### Linear Regression

In this section we will perform a simple linear regression. 

First we want to initialize a new Node project and install our dependencies. For this section of this chapter we will be using the `@tensorflow/tfjs` library, which is the TFJS bindings for Node.js. This will allow us to take advance of low-level implementations in C to perform complex linear algebra operations for us. This version of the package runs these calculations on the CPU rather than GPU and will be slower than the `@tensorflow/tfjs-gpu` version which runs calculations on the GPU. However, since not every read will have access to a GPU that is compatible with TFJS we will use this version.  However, if you wish to train your models on a GPU then feel free to use the GPU version.

First we will need to create a new directory to hold our TFJS model. 

```bash
mkdir regression
cd regression
```
This "regression" directory will host our TFJS NPM package. To initialize the directory as a new npm package and accept the system defaults run the following command.

```bash
npm init -f
```

This will create a package.json file that will look something like this.

```javascript
{
  "name": "regression",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [],
  "author": "",
  "license": "ISC"
}

```

The package.json file is a special file that contains metadata that describes our Node application. In this case the system provided defaults that describe the package name, version, description, main entry point (index.js), test script, keywords, author and license. We will change some of the defaults to make our package.json file a little more descriptive by adding a description and adding myself as the author.

```javascript
{
  "name": "regression",
  "version": "1.0.0",
  "description": "A simple linear regression model written using TFJS",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [],
  "author": "Jamal O'Garro",
  "license": "ISC"
}
```

The next step is to add our package manager, Yarn in our case, to allow us to manage our dependencies. Let's get this installed via the NPM CLI.

```bash
npm install -D yarn
```

Next we want to add our main dependencies, which are TensorFlow.js and the Babel, which is a compiler for JavaScript that will allow us to take advantage of several next generation JavaScript features not currently available in Node such as the ability to use ES6 style module imports. We can do this by running the following command.

```bash
npx yarn add @tensorflow/tfjs-node @babel/cli @babel/preset-env @babel/cli @babel/core @babel/preset-es2015 babel-polyfill
```

This command will use Yarn to install our dependencies, save them into a local directory call `node_modules` and add new "dependencies" section to our package.json file. If you inspect your package.json file it should look similar to this.

```JavaScript
{
  "name": "regression",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [],
  "author": "",
  "license": "ISC",
  "devDependencies": {
    "yarn": "^1.13.0"
  },
  "dependencies": {
    "@babel/cli": "^7.2.3",
    "@babel/core": "^7.4.0",
    "@babel/preset-env": "^7.4.2",
    "@babel/preset-es2015": "^7.0.0-beta.53",
    "@tensorflow/tfjs-node": "^1.0.2",
    "babel-polyfill": "^6.26.0"
  }
}
```

Next let's make a directory to contain our compiled JavaScript.

```bash
mkdir compiled
```

Next we'll add an NPM script that will compile our ES6 code using Babel, output the compiled result into our `compiled` directory and run the compiled code. We do this by adding a new property in the "scripts" section of our package.json file and add the following code "npx babel regression.js --out-file compiled/regression.js && node compiled/regression.js". We should also move the "test" script since we won't be using it at all. After making these changes your package.json file should look like the following example.

```javascript
{
  "name": "regression",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "run-regression": "npx babel regression.js --out-file compiled/regression.js && node compiled/regression.js"
  },
  "keywords": [],
  "author": "",
  "license": "ISC",
  "devDependencies": {
    "yarn": "^1.13.0"
  },
  "dependencies": {
    "@babel/cli": "^7.2.3",
    "@babel/core": "^7.4.0",
    "@babel/preset-env": "^7.4.2",
    "@babel/preset-es2015": "^7.0.0-beta.53",
    "@tensorflow/tfjs-node": "^1.0.2",
    "babel-polyfill": "^6.26.0"
  }
}
```

Before we can run our model we have to add one last file to our directory, which is to specify instructions to Babel so it knows how to compile our JavaScript for node. We do this by adding a .babelrc file in the root of our directory.

```bash
touch .babelrc
```

Next we add the following contents to this file and save it. Don't worry too much about the contents of this file. It's just a way to let Babel know how to properly compile our next generation JavaScript to a version that Node can understand. 

```javascript
{
  "presets": [
    [
      "@babel/preset-env",
      {
        "useBuiltIns": "entry"
      }
    ]
  ]
}
```

Next let's create a new file that will be used to contain our regression model. We'll call this file "regression.js".

```bash
touch regression.js
```

And inside of `regression.js` let's add a simple example to test our environment setup.

```JavaScript
import * as tf from "@tensorflow/tfjs-node";
import "babel-polyfill";

const main = async() => {
  const helloWorld = tf.scalar('Hello TensorFlow.js');
  helloWorld.print();
};

if (require.main === module)
  main();
```
In the example above we are importing the entire `@tensorflow/tfjs-node` library and storing a reference to it inside of a variable called tf. We also import the `babel-polyfil` so that we can use Babel. We also define a main function that will be used to run our "hello world" example. This is anonymous async function, which will allow us to write asynchronous TFJS code in a synchronous style when we start working with parts of the library that return promises. Inside of main we create an instance of `tf.scalar` containing the string "Hello TensorFlow.js". We then use the `print` method on this instance to print this value to the screen. Last we add a main guard, to execute the main function only if our program is being executed from the command line. Now let's run it using the NPM script that we defined inside of our package.json file.

```bash
npx yarn run-regression

Tensor
    Hello TensorFlow.js
✨  Done in 1.81s.
```

I appears that our script is running as expected and we can now begin writing our model. We shall begin by creating our input and output data. For this we will make use of the `tf.linspace` method that will create a 1D tensor over some range in equal size steps. In our example we will create X's ranging from 0 to 10 in steps of 10 to be used as our input. Our output will consist of numbers from 10 to 11 in equal steps of 10. Let's define and print our values as follows.

```JavaScript
const main = async() => {
  // create random data
  const xs = tf.linspace(0, 10, 10);
  const ys = tf.linspace(10, 11, 10);

  xs.print();
  ys.print();
}
```

Now run the the model to see the results.

```bash
npx yarn run-regression

Tensor
    [0, 1.1111112, 2.2222223, 3.3333335, 4.4444447, 5.5555558, 6.666667, 7.7777781, 8.8888893, 10]
Tensor
    [10, 10.1111107, 10.2222214, 10.3333321, 10.4444427, 10.5555534, 10.6666641, 10.7777748, 10.8888855, 10.9999962]
```

Our next step will be to create a regression formula that we will use to predict a value of an output y from an input x. In regression we usually use a formula that is defined as follows:

> $$ y = \beta_{0} + \beta_{1} * x + \epsilon$$

Where B0 is the y-intercept, x is the independent variable, w is a weight or slope of our regression line, y-hat is the estimated value and epsilon is some error term that describes noise in our distribution. We easily define this function using TensorFlow.js. In the field of machine learning we usually refer to the y-intercept as the bias term and the slope as our weight.

```javascript
const main = async() => {
  ...
    // create random variables for our slope and intercept values
  const m = tf.scalar(Math.random()).variable();
  const b = tf.scalar(Math.random()).variable();

  // regression formula y^ = m*x + b
  const func = x => x.mul(m).add(b);
}
```

Our machine model will be trained when we perform the steps required to find the bias and weight that provides the least amount of error. To minimize the error described by epsilon, we will define a cost function that measures the size of this error. In regression we typically want to reduce the mean squared error or MSE, which is defined by the following formula.

> $$ MSE = \Sigma (\hat{y} - y)^2 / n $$

This can be written in TFJS as follows.

```javascript
const main = async() => {
  ...
  // MSE loss function
  const loss = (yHat, y) => yHat.sub(y).square().mean();
}
```

Our next step is to minimize this loss by performing backpropogation. For we'll use stochastic gradient descent o minimize the cost of our regression function by training it over 500 epochs with a learning rate of 0.01.

```javascript
const main = async() => {
  ...
  // create our Stochastic Gradient Descent optimizer from the previous chapter
  const learningRate = 0.01;
  const optimizer = tf.train.sgd(learningRate);
  
  // Train the model.
  const epochs = 500;
  for (let i = 0; i < epochs; i++) {
    optimizer.minimize(() => loss(func(xs), ys));
  }
}
```

Once we train our model we can use it perform inference.

```javascript
  // perform inference by passing a 1d tensor to our trined function
  const xValue = tf.tensor1d([1]);
  const prediction = await func(xValue).data();

  // print our learned slope and intercept
  m.print();
  b.print();
  console.log(prediction);
```

When we run our model we get the following output.

```javascript
npx yarn run-regression

Tensor
    0.17935864627361298
Tensor
    9.446050643920898
Float32Array [ 9.625409126281738 ]
```

Our model was able to back our a value of roughly 0.1794 for our weight and 9.446 for our bias-term. We test the predictive power of our model by passing in values to see if the output is what we expect it to be. Inputting one yields 9.625, which is roughly equal to ~9.625 the result of 1 * 9.446050643920898 + 0.17935864627361298 so we can say that our model does a somewhat descent job of predicting values. There are more rigorous ways to test the strength of our model; however, we will save that for the next chapter since we will be able to rely on several summary tools provided to us by the TFJS Layers API.


#### Logistic Regression

Logistic Regression is a form of regression that is used to classify data into one of two categories and provides the solution to a binary problem. 

TBD: Add Example


## Summary

In this chapter we took a deep into the TensorFlow.js Core API and learned how we can create different tensors, perform operations on these tensors, how to build an ANN, how to train and test the accuracy of this ANN. In the next chapter we will build upon what we learned in this chapter to obtain a better understanding the the higher-level Layers API, which is built directly on top of the Core API covered in this chapter. The emphasis was placed on the most common components of the Core API that you will use on a day-to-day basis. The data structures, methods and operations that we covered in this chapter only scratch the surface and to find get an idea of what else is available in the core I recommend spending some time reading through the TensorFlow.js documentation and if you’re feeling really adventurous the source code is worth a gander as well.
