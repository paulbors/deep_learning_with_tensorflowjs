"use strict";

var tf = _interopRequireWildcard(require("@tensorflow/tfjs-node"));

require("babel-polyfill");

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = Object.defineProperty && Object.getOwnPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : {}; if (desc.get || desc.set) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } } newObj.default = obj; return newObj; } }

function asyncGeneratorStep(gen, resolve, reject, _next, _throw, key, arg) { try { var info = gen[key](arg); var value = info.value; } catch (error) { reject(error); return; } if (info.done) { resolve(value); } else { Promise.resolve(value).then(_next, _throw); } }

function _asyncToGenerator(fn) { return function () { var self = this, args = arguments; return new Promise(function (resolve, reject) { var gen = fn.apply(self, args); function _next(value) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "next", value); } function _throw(err) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "throw", err); } _next(undefined); }); }; }

// Figure 6.1
var model = tf.sequential();
model.add(tf.layers.dense({
  units: 32,
  inputShape: [100]
}));
model.add(tf.layers.dense({
  units: 4
})); // Function to execute model examples

var predict =
/*#__PURE__*/
function () {
  var _ref = _asyncToGenerator(
  /*#__PURE__*/
  regeneratorRuntime.mark(function _callee() {
    var model, xs, ys;
    return regeneratorRuntime.wrap(function _callee$(_context) {
      while (1) {
        switch (_context.prev = _context.next) {
          case 0:
            // Figure 6.2
            // create model and add a dense layer
            model = tf.sequential();
            model.add(tf.layers.dense({
              units: 1,
              inputShape: [1]
            })); // compile the model

            model.compile({
              loss: "meanSquaredError",
              optimizer: "sgd"
            }); // traning data

            xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
            ys = tf.tensor2d([1, 3, 5, 7], [4, 1]); // train model with our training data

            _context.next = 7;
            return model.fit(xs, ys);

          case 7:
            // use the model to predict an output
            model.predict(tf.tensor2d([5], [1, 1])).print(); // Figure 6.3

            model.summary(); // Figure 6.4

            model.summary(30, [100, 50, 25], function (x) {
              return console.log("output: " + x);
            });

          case 10:
          case "end":
            return _context.stop();
        }
      }
    }, _callee, this);
  }));

  return function predict() {
    return _ref.apply(this, arguments);
  };
}(); // invoke our predict function


predict();
