/*
 * Chapter 5 ANN example
 *
 * This file contains the "real-world" ANN example that we built using the core
 * API in chapter 5. The code in this file can be run using the "npx yarn"
 * run-ann-example" command.
 *
 */

import * as tf from "@tensorflow/tfjs-node";
import "babel-polyfill";

const BOSTON_HOUSING_CSV_URL =
  "https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv";
const CSV_DATA_CONFIG = {
  columnConfigs: {
    medv: {
      isLabel: true
    }
  }
};
const LEARNING_RATE = 0.000001;
const BATCH_SIZE = 10;
const EPOCS = 10;
const LOSS_FUNCTION = "meanSquaredError";

// const STEPS = 1000;
// const PRICE_NORM_FACTOR = 1000;

const buildAndTrainModel = async () => {
  const csvDataset = tf.data.csv(BOSTON_HOUSING_CSV_URL, CSV_DATA_CONFIG);
  // print the config
  // console.log("HOUSING DATA: ", csvDataset);

  // number of features
  const columns = await csvDataset.columnNames();
  const numOfFeatures = columns.length - 1;
  console.log("COLUMN NAMES: ", columns, numOfFeatures);

  // prepare the data set for training
  const flattenedDataset = csvDataset
    .map(([rawFeatures, rawLabel]) => [
      Object.values(rawFeatures),
      Object.values(rawLabel)
    ])
    .batch(BATCH_SIZE);

  // Define the model
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [numOfFeatures],
      units: 1
    })
  );

  // compile the model
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: LOSS_FUNCTION
  });

  return model.fitDataset(flattenedDataset, {
    epochs: EPOCS,
    callbacks: {
      onEpochEnd: async (epoc, logs) => {
        console.log(`${epoc} : ${logs.loss}`);
      }
    }
  });
};

buildAndTrainModel();
