/*
 * Chapter 5 code examples
 *
 * Since these examples simply log output to the screen they can be run in the
 * terminal using the command "yarn run-examples". To see individual outputs
 * simply comment the lines you do not wish to run.
 *
 * Be sure to install the dependencies by running "npx yarn" before trying to
 * run the examples script.
 *
 */

import * as tf from "@tensorflow/tfjs-node";

// Figure 5.1
const vector = tf.tensor([1, 2, 3, 4, 5]);
vector.print();

// Figure 5.2

const matrix = tf.tensor([[0, 0, 0], [1, 1, 1]]);
matrix.print();

// Figure 5.3
const matrix2 = tf.tensor([0, 0, 0, 1, 1, 1], [2, 3]);
matrix2.print(); // => Tensor [[0, 0, 0], [1, 1, 1]]

// Figure 5.4
const int32Matrix = tf.tensor([0, 0, 0, 1, 1, 1], [2, 3], "int32");
int32Matrix.print(); // => Tensor[[0, 0, 0],[1, 1, 1]]
int32Matrix.dtype; // => ‘int32’

// Figure 5.5
const scalar = tf.scalar(3.14);
scalar.print(); // => Tensor; 3.140000104904175;

// Figure 5.6
const matrix1d = tf.tensor1d([3, 2, 1]);
matrix1d.print();

// Figure 5.7
const matrix2d = tf.tensor2d([5, 4, 3, 2, 1, 0], [3, 2]);
matrix2d.print();

// reshape
matrix2d.reshape([2, 3]).print();

matrix2d.reshape([1, 6]).print();

matrix2d.reshape([6, 1]).print();

// Figure 5.x

const ceilTensor = tf.tensor1d([0.3]);

// Figure 5.x
const absTensor = tf.tensor1d([-10, -20, -30, -40, -50]);
absTensor.abs().print();
