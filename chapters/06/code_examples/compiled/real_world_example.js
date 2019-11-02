"use strict";

var tf = _interopRequireWildcard(require("@tensorflow/tfjs-node"));

var argparse = _interopRequireWildcard(require("argparse"));

require("babel-polyfill");

var _data = _interopRequireDefault(require("./data"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = Object.defineProperty && Object.getOwnPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : {}; if (desc.get || desc.set) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } } newObj.default = obj; return newObj; } }

function asyncGeneratorStep(gen, resolve, reject, _next, _throw, key, arg) { try { var info = gen[key](arg); var value = info.value; } catch (error) { reject(error); return; } if (info.done) { resolve(value); } else { Promise.resolve(value).then(_next, _throw); } }

function _asyncToGenerator(fn) { return function () { var self = this, args = arguments; return new Promise(function (resolve, reject) { var gen = fn.apply(self, args); function _next(value) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "next", value); } function _throw(err) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "throw", err); } _next(undefined); }); }; }

// define our model
var model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  filters: 32,
  kernelSize: 3,
  activation: "relu"
}));
model.add(tf.layers.conv2d({
  filters: 32,
  kernelSize: 3,
  activation: "relu"
}));
model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2]
}));
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: "relu"
}));
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: "relu"
}));
model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2]
}));
model.add(tf.layers.flatten());
model.add(tf.layers.dropout({
  rate: 0.25
}));
model.add(tf.layers.dense({
  units: 512,
  activation: "relu"
}));
model.add(tf.layers.dropout({
  rate: 0.5
}));
model.add(tf.layers.dense({
  units: 10,
  activation: "softmax"
}));
var optimizer = "rmsprop";
model.compile({
  optimizer: optimizer,
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"]
});

var run =
/*#__PURE__*/
function () {
  var _ref = _asyncToGenerator(
  /*#__PURE__*/
  regeneratorRuntime.mark(function _callee(epochs, batchSize, modelSavePath) {
    var _data$getTrainData, trainImages, trainLabels, epochBeginTime, millisPerStep, validationSplit, numTrainExamplesPerEpoch, numTrainBatchesPerEpoch, _data$getTestData, testImages, testLabels, evalOutput;

    return regeneratorRuntime.wrap(function _callee$(_context) {
      while (1) {
        switch (_context.prev = _context.next) {
          case 0:
            _context.next = 2;
            return _data.default.loadData();

          case 2:
            // get our training data
            _data$getTrainData = _data.default.getTrainData(), trainImages = _data$getTrainData.images, trainLabels = _data$getTrainData.labels; // print the model summary

            model.summary();
            validationSplit = 0.15;
            numTrainExamplesPerEpoch = trainImages.shape[0] * (1 - validationSplit);
            numTrainBatchesPerEpoch = Math.ceil(numTrainExamplesPerEpoch / batchSize);
            _context.next = 9;
            return model.fit(trainImages, trainLabels, {
              epochs: epochs,
              batchSize: batchSize,
              validationSplit: validationSplit
            });

          case 9:
            _data$getTestData = _data.default.getTestData(), testImages = _data$getTestData.images, testLabels = _data$getTestData.labels;
            evalOutput = model.evaluate(testImages, testLabels);
            console.log("\nEvaluation result:\n" + "  Loss = ".concat(evalOutput[0].dataSync()[0].toFixed(3), "; ") + "Accuracy = ".concat(evalOutput[1].dataSync()[0].toFixed(3)));

            if (!(modelSavePath != null)) {
              _context.next = 16;
              break;
            }

            _context.next = 15;
            return model.save("file://".concat(modelSavePath));

          case 15:
            console.log("Saved model to path: ".concat(modelSavePath));

          case 16:
          case "end":
            return _context.stop();
        }
      }
    }, _callee, this);
  }));

  return function run(_x, _x2, _x3) {
    return _ref.apply(this, arguments);
  };
}();

var parser = new argparse.ArgumentParser({
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
var args = parser.parseArgs();
console.log("\nModel run with the following arguments:\n---------------------------------------\nEpochs: ".concat(args.epochs, "\nBatch Size: ").concat(args.batch_size, "\nModel Save Path: ").concat(args.model_save_path, "\n"));
tf.disableDeprecationWarnings();
run(args.epochs, args.batch_size, args.model_save_path);
