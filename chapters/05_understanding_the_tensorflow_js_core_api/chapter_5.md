# Chapter 5 - Understanding the TensorFlow.js Core API

In the last chapter we build our very first Artificial Neural Network (ANN) using the TensorFlow.js layers API. In this chapter we will take a dive into the TensorFlow.js Core API, which the Layers API that we received a brief introduction to is build on top of. Our aim is to explore this low-level API to learn how it is structured as well as what methods and data structures it provides to allow us to create deep learning models that can be run in the browser. By the end of this chapter we should:

- Understand the role of tensors
- Be able to differentiate between scalar, vector and matrix tensors
- How to perform operations on tensors
- Build an Artificial Neural Network from scratching using the Core API
- How to train an ANN and predict results
- How to test the mode on unseen data

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

By default, the numbers that are used to create our vectors are 32-bit floats. However, what happens when we want to use 32-bit integers instead? Luckily for us the solution is simple since we can easily specify the datatype of our values by passing a third argument to the `t​f.tensor​utilityfunction`.

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
- **rank**​ - specifies the number of dimensions of the tf.Tensor object. The terms order, degree and n-dimension are also used. Each TensorFlow.js rank corresponds to a particular mathe entity, as demonstrated in the table below:

![](images/05_table.png)

- **rankType**​ - represents the rank type of an array, which can be any of the values ‘R0’, ‘R1’, ‘R2’, ‘R3’, ‘R4’, ‘R5’ or ‘R6’.
- **shape** -​ represents the shape of the tensor, which is the number of elements in each dimension. Another way to view this is as how many columns and rows are contained within it. For example, a matrix consisting of two columns and five elements in each column would have a shape of 2x5.
- **size**​ - specifies the number of elements in a tensor. This is typically the result of multiplying the number of columns by the number of elements in each column. So for for example, a 3x3 matrix would have a size of 9.
- **strides**​ - represents the number of elements to skip in each dimension when indexing.

_A Note on Mathematical Explanation of the terms Scalar, Vector, one-dimensional and N-dimensional. (optional section)_

In this chapter we will use a lot of terms that come from mathematics, more specifically linear algebra. Some readers may have little linear algebra experience or it may have been years since they completed a college course on the subject so we will briefly define these terms below. This section is optional but may be useful since these terms will come up again later in this chapter and throughout the rest of the book.

- Scalar - TBD
- Vector - TBD
- Matrix - TBD
- Rank - TBD
- Size - TBD
- Shape - TBD

You may be wondering why TensorFlow uses vectors to perform machine learning tasks it primarily because the operations that can be performed on these structures allows us to train models with a high degree of efficiency. Later in this chapters we’ll see how matrix multiplication can help us perform complex calculations in very few steps. Since just about any real-world data can be represented as a vector this is good news for us as machine learning engineers. If you would like to take a deeper dive into Linear Algebra I would recommend reviewing a textbook on the subject to gain a deeper understanding.

### Tensor Types

This section will explain the different Tensor types that are provided by the TensorFlow API. Though some of these concepts are derived from the subject of Linear Algebra we will keep the math involved to a minimum and will focus primarily on the code.

The different types of tensors that we will typically see in our machine learning models are as follows:

- **Scalar**​ - a tensor of rank 0. These values can be thought of as single numbers that are not contained in an array, like the number 33 for example.
- **Vector​** - a tensor of rank 1. This can be thought of as a tensor that contains an array of one column. We also refer to this as a one-dimensional or 1D tensor.
- **2D Matrix** ​- a tensor of rank 2. This can be thought of as a tensor that contains an containing two arrays of values. We also refer to this as a two-dimensional or 2D tensor.
- **3D Matrix**​ - a tensor of rank 3. This can be thought of as a tensor that contains an containing three arrays of values. We also refer to this as a two-dimensional or 3D tensor.
- **4D Matrix**​ - a tensor of rank 4. This can be thought of as a tensor that contains an containing four arrays of values. We also refer to this as a two-dimensional or 4D tensor.

### Utility Methods for Creating Tensor Types

In the previous section we took a look at some of the different types of tensors provided to us by TensorFlow.js. We also took a look at how we can create tensors of different shapes using the `t​f.tensor​utilityfunction`. However, TensorFlow.js provides additional utility functions that make it convenient for us to create different types of tensors in one line of code.

####Creating Scalars Tensors

To easily create a scalar tensor, which is a tensor of rank 0, we can use the `tf.scalar` utility method. It similar to `tf.tensor` in that it returns a `tf.Tensor` (actually a `tf.Scalar`) object; however, the shape is implied by the function name and all we have to do to create a scalar value is pass the value and data type to this function.

Figure 5.5

```javascript
const scalar = tf.scalar(3.14);
scalar.print(); // =>
Tensor;
3.140000104904175;
```

#### Creating 1D Tensors

To create a one-dimensional (1D) tensor the TensorFlow.js library provides us with another utility method called `tf.tensor1d`. Though we can create a tensor of one dimension using `tf.tensor()` it is recommended to use `tf.tensor1d()` to be explicit and make our code more readable.

**Figure 5.6**

```JavaScript
const matrix1d = tf.tensor1d([3, 2, 1]);
matrix1d.print();

/*
Tensor
    [3, 2, 1]
*/
```

#### Creating 2D Tensors

Similar to how we can create a one-dimensional tensor using `tf.tensor1d` we can create a two-dimensional tensor using the `tf.tensor2d` method.

**Figure 5.7**

```JavaScript
const matrix2d = tf.tensor2d([5, 4, 3, 2, 1, 0], [3, 2]);
matrix2d.print();

/*
Tensor
    [[5, 4],
     [3, 2],
     [1, 0]]
*/
```

#### Creating 3D Tensors

If we want to create a tensor that is of three dimensions we can make use of the `tf.tensor3d` method provided by the TFJS library.

**Figure 5.8**

```JavaScript

```

#### Creating Tensors of Different Dimension

TensorFlow.js provides helper methods that will alow us to create matrices of up to six dimensions. They are `tf.tensor4d`, `tf.tensor5d` and `tf.tensor6d`. They work similar in fashion to the three methods we previously looked at. Feel free to experiment with them.

### Performing Operations on Tensors

When building our deep learning models we often have to perform operations on our tensors. These operations are usually mathematical in nature and involve adding, subtracting, multiplying and dividing tensors.

#### Math Operations

We can also

#### tf.abs

In order to compute the absolute value of all values in a tensor we can use the `tf.abs` method.

```JavaScript
const absTensor = tf.tensor1d([-10, -20, -30, -40, -50]);
absTensor.abs().print();

/*
Tensor
    [10, 20, 30, 40, 50]
*/
```

#### tf.ceil

TBD

**Figure 5.x**

```JavaScript

```

#### tf.reshape

When working with tensors we may need to reshape, to a 2x3 matrix from a 3x2 matrix for example. In order to achieve this we can leverage the `reshape` method that exists on tensor instances. One trick to keep in mind is that we can reshape a tensor to any dimension as long as the product remains the same. Figure 5.x demonstrates how we can reshape using the `reshape` method that exists on tensors.

**Figure 5.x**

```JavaScript
// create a two-dimensional tensor
const matrix2d = tf.tensor2d([5, 4, 3, 2, 1, 0], [3, 2]);
matrix2d.print();
/*
Tensor
    [[5, 4],
     [3, 2],
     [1, 0]]
*/

matrix2d.reshape([2, 3]).print();
/*
Tensor
    [[5, 4, 3],
     [2, 1, 0]]
*/

matrix2d.reshape([1, 6]).print();
/*
Tensor
     [[5, 4, 3, 2, 1, 0],]
*/

matrix2d.reshape([6, 1]).print();

/*
Tensor
    [[5],
     [4],
     [3],
     [2],
     [1],
     [0]]
*/
```

**Figure 5.x**

```JavaScript

```

### Building a Simple ANN Using the Core API

**Figure 5.x**

```JavaScript

```

## Summary

In this chapter we took a deep into the TensorFlow.js Core API and learned how we can create different tensors, perform operations on these tensors, how to build an ANN, how to train and test the accuracy of this ANN. In the next chapter we will build upon what we learned in this chapter to obtain a better understanding the the higher-level Layers API, which is built directly on top of the Core API covered in this chapter. The emphasis was placed on the most common components of the Core API that you will use on a day-to-day basis. The data structures, methods and operations that we covered in this chapter only scratch the surface and to find get an idea of what else is available in the core I would recommend spending some time reading through the TensorFlow.js documentation and if you’re feeling really adventurous the source code is worth a gander as well.
