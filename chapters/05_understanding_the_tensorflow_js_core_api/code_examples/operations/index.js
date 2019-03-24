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
import "babel-polyfill";


const main = async() => {
  const scalar = tf.scalar(3.14);
  scalar.print(); 

  const absTensor = tf.tensor1d([-10, -20, -30, -40, -50]).abs();
  absTensor.print();

  const range = tf.range(0, 100, 10);
  range.print();


  const filled = tf.fill([2, 3], 5);
  filled.print();

  const ones = tf.ones([4, 4]);
  ones.print();

  const zeros = tf.zeros([5, 5]);
  zeros.print();

  const ceiled = tf.tensor1d([.6, 1.1, -3.3]).ceil();
  ceiled.print();

  const floored = tf.tensor1d([.6, 1.1, -3.3]).floor();
  floored.print();

  const spaced = tf.linspace(0, 1, 10)
  spaced.print();

  const eye = tf.eye(5);
  eye.print();

  const eye10 = tf.eye(10);
  eye10.print();
};

if (require.main === module)
  main();