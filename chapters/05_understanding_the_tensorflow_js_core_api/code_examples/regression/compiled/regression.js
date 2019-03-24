"use strict";

var tf = _interopRequireWildcard(require("@tensorflow/tfjs-node"));

require("babel-polyfill");

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = Object.defineProperty && Object.getOwnPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : {}; if (desc.get || desc.set) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } } newObj.default = obj; return newObj; } }

function asyncGeneratorStep(gen, resolve, reject, _next, _throw, key, arg) { try { var info = gen[key](arg); var value = info.value; } catch (error) { reject(error); return; } if (info.done) { resolve(value); } else { Promise.resolve(value).then(_next, _throw); } }

function _asyncToGenerator(fn) { return function () { var self = this, args = arguments; return new Promise(function (resolve, reject) { var gen = fn.apply(self, args); function _next(value) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "next", value); } function _throw(err) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "throw", err); } _next(undefined); }); }; }

var print = function print(x) {
  return x.print();
};

var main =
/*#__PURE__*/
function () {
  var _ref = _asyncToGenerator(
  /*#__PURE__*/
  regeneratorRuntime.mark(function _callee() {
    var xs, ys, m, b, func, loss, learningRate, optimizer, epochs, i, xValue, prediction;
    return regeneratorRuntime.wrap(function _callee$(_context) {
      while (1) {
        switch (_context.prev = _context.next) {
          case 0:
            // create random data
            xs = tf.linspace(0, 10, 10);
            ys = tf.linspace(10, 11, 10);
            xs.print();
            ys.print(); // create random variables for our slope and intercept values

            m = tf.scalar(Math.random()).variable();
            b = tf.scalar(Math.random()).variable(); // regression formula y^ = m*x + b

            func = function func(x) {
              return x.mul(m).add(b);
            }; // MSE loss function


            loss = function loss(yHat, y) {
              return yHat.sub(y).square().mean();
            }; // define our Stochastic Gradient Descent optimizer from the previous chapter


            learningRate = 0.01;
            optimizer = tf.train.sgd(learningRate); // Train the model.

            epochs = 500;

            for (i = 0; i < epochs; i++) {
              optimizer.minimize(function () {
                return loss(func(xs), ys);
              });
            } // perform inference


            xValue = tf.tensor1d([1]);
            _context.next = 15;
            return func(xValue).data();

          case 15:
            prediction = _context.sent;
            m.print();
            b.print();
            console.log(prediction);

          case 19:
          case "end":
            return _context.stop();
        }
      }
    }, _callee);
  }));

  return function main() {
    return _ref.apply(this, arguments);
  };
}();

if (require.main === module) main();
