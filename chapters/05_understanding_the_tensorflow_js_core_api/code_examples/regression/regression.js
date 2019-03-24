/**
 * Simple Linear Regression in TFJS
 */
import * as tf from "@tensorflow/tfjs-node";
import "babel-polyfill";

const print = x => x.print();

const main = async() => {
  // create random data
  const xs = tf.linspace(0, 10, 10);
  const ys = tf.linspace(10, 11, 10);

  xs.print();
  ys.print();

  // create random variables for our slope and intercept values
  const m = tf.scalar(Math.random()).variable();
  const b = tf.scalar(Math.random()).variable();

  // regression formula y^ = m*x + b
  const func = x => x.mul(m).add(b);

  // MSE loss function
  const loss = (yHat, y) => yHat.sub(y).square().mean();

  // define our Stochastic Gradient Descent optimizer from the previous chapter
  const learningRate = 0.01;
  const optimizer = tf.train.sgd(learningRate);
  
  // Train the model.
  const epochs = 500;
  for (let i = 0; i < epochs; i++) {
    optimizer.minimize(() => loss(func(xs), ys));
  }

  // perform inference
  const xValue = tf.tensor1d([1]);
  const prediction = await func(xValue).data();
  m.print();
  b.print();
  console.log(prediction);
};

if (require.main === module)
  main();