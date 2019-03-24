/*
 * Chapter 5 real world example
 *
 * Since these examples simply log output to the screen they can be run in the
 * terminal using the command "yarn real-world-example". To see individual outputs
 * simply comment the lines you do not wish to run.
 *
 * Be sure to install the dependencies by running "npx yarn" before trying to
 * run the examples script.
 *
 * Note: This example comes directly from the TensorFlow.js documentation but has
 * been explained in further detail in chatper 5 of this publication.
 *
 */

import * as tf from "@tensorflow/tfjs-node";
import "babel-polyfill";

// create our placeholder variables that we want to predict
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

// create out optimization function
const EPOCHS = 200;
const LEARNING_RATE = 0.5;
const optimizer = tf.train.sgd(LEARNING_RATE);

// helper function that will generate our data
const generateData = (numPoints, coeff, sigma = 0.04) => {
  return tf.tidy(() => {
    const [a, b, c, d] = [
      tf.scalar(coeff.a),
      tf.scalar(coeff.b),
      tf.scalar(coeff.c),
      tf.scalar(coeff.d)
    ];

    const xs = tf.randomUniform([numPoints], -1, 1);

    const three = tf.scalar(3, "int32");
    const ys = a
      .mul(xs.pow(three))
      .add(b.mul(xs.square()))
      .add(c.mul(xs))
      .add(d)
      .add(tf.randomNormal([numPoints], 0, sigma));

    const ymin = ys.min();
    const ymax = ys.max();
    const yrange = ymax.sub(ymin);
    const ysNormalized = ys.sub(ymin).div(yrange);

    return {
      xs,
      ys: ysNormalized
    };
  });
};

// define the function that we want to predict coefficients for (a * x^3 + b*x^2 + c*x + d)
const predict = x => {
  return tf.tidy(() => {
    return a
      .mul(x.pow(tf.scalar(3, "int32")))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d);
  });
};

// define our loss function (MSE)
const loss = (prediction, labels) => {
  // (y-hat - y)^2 / n
  const error = prediction
    .sub(labels)
    .square()
    .mean();
  return error;
};

// train our model using backpropagation to minimize our loss function
const train = async (xs, ys, epochs) => {
  for (let i = 0; i < epochs; i++) {
    optimizer.minimize(() => {
      // Feed the examples into the model
      const pred = predict(xs);
      return loss(pred, ys);
    });
  }
};

// learn the coefficients
const learnCoefficients = async () => {
  const trueCoefficients = { a: -0.8, b: -0.2, c: 0.9, d: 0.5 };
  const trainingData = generateData(100, trueCoefficients);

  const predictionsBefore = predict(trainingData.xs);

  // log the coefficients before training
  console.log("a before training: ", a.dataSync()[0]);
  console.log("b before training: ", b.dataSync()[0]);
  console.log("c before training: ", c.dataSync()[0]);
  console.log("d before training: ", d.dataSync()[0]);

  // train the model
  await train(trainingData.xs, trainingData.ys, EPOCHS);

  const predictionsAfter = predict(trainingData.xs);

  // log the coefficients after training
  console.log("a after training: ", a.dataSync()[0]);
  console.log("b after training: ", b.dataSync()[0]);
  console.log("c after training: ", c.dataSync()[0]);
  console.log("d after training: ", d.dataSync()[0]);

  // clean up our data
  predictionsBefore.dispose();
  predictionsAfter.dispose();
};

learnCoefficients();
