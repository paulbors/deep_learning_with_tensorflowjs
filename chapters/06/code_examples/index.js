/*
 * Chapter 6 code examples
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

// Figure 6.1
const model = tf.sequential();
model.add(tf.layers.dense({ units: 32, inputShape: [100] }));
model.add(tf.layers.dense({ units: 4 }));

// Function to execute model examples
const predict = async () => {
  // Figure 6.2
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

  // Figure 6.3
  model.summary();

  // Figure 6.4
  model.summary(30, [100, 50, 25], x => console.log("output: " + x));
};

// invoke our predict function
predict();
