/*
 * Chapter 6 real world example
 *
 * Since these examples simply log output to the screen they can be run in the
 * terminal using the command "npx yarn real-world-example". To see individual outputs
 * simply comment the lines you do not wish to run.
 *
 * Be sure to install the dependencies by running "npx yarn" before trying to
 * run the examples script.
 *
 * Note: This example comes directly from the TensorFlow.js documentation but has
 * been explained in further detail in chatper 6 of this publication.
 *
 */

import * as tf from "@tensorflow/tfjs-node-gpu";
import * as argparse from "argparse";
import "babel-polyfill";
import data from "./data";

// define our model
const model = tf.sequential();
model.add(
  tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3,
    activation: "relu"
  })
);
model.add(
  tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    activation: "relu"
  })
);
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(
  tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: "relu"
  })
);
model.add(
  tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: "relu"
  })
);
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(tf.layers.flatten());
model.add(tf.layers.dropout({ rate: 0.25 }));
model.add(tf.layers.dense({ units: 512, activation: "relu" }));
model.add(tf.layers.dropout({ rate: 0.5 }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

const optimizer = "rmsprop";
model.compile({
  optimizer: optimizer,
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"]
});

const run = async (epochs, batchSize, modelSavePath) => {
  // load our data
  await data.loadData();

  // get our training data
  const { images: trainImages, labels: trainLabels } = data.getTrainData();

  // print the model summary
  model.summary();

  let epochBeginTime;
  let millisPerStep;
  const validationSplit = 0.15;
  const numTrainExamplesPerEpoch = trainImages.shape[0] * (1 - validationSplit);
  const numTrainBatchesPerEpoch = Math.ceil(
    numTrainExamplesPerEpoch / batchSize
  );
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit
  });

  const { images: testImages, labels: testLabels } = data.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(
    `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`
  );

  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
};

const parser = new argparse.ArgumentParser({
  description: "TensorFlow.js-Node MNIST Example.",
  addHelp: true
});
parser.addArgument("--epochs", {
  type: "int",
  defaultValue: 20,
  help: "Number of epochs to train the model for."
});
parser.addArgument("--batch_size", {
  type: "int",
  defaultValue: 128,
  help: "Batch size to be used during model training."
});
parser.addArgument("--model_save_path", {
  type: "string",
  defaultValue: "./trained_models",
  help: "Path to which the model will be saved after training."
});
const args = parser.parseArgs();

console.log(`
Model run with the following arguments:
---------------------------------------
Epochs: ${args.epochs}
Batch Size: ${args.batch_size}
Model Save Path: ${args.model_save_path}
`);

tf.disableDeprecationWarnings();

run(args.epochs, args.batch_size, args.model_save_path);
