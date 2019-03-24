"use strict";

var tf = _interopRequireWildcard(require("@tensorflow/tfjs-node"));

require("babel-polyfill");

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) { var desc = Object.defineProperty && Object.getOwnPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : {}; if (desc.get || desc.set) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } } newObj.default = obj; return newObj; } }

function asyncGeneratorStep(gen, resolve, reject, _next, _throw, key, arg) { try { var info = gen[key](arg); var value = info.value; } catch (error) { reject(error); return; } if (info.done) { resolve(value); } else { Promise.resolve(value).then(_next, _throw); } }

function _asyncToGenerator(fn) { return function () { var self = this, args = arguments; return new Promise(function (resolve, reject) { var gen = fn.apply(self, args); function _next(value) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "next", value); } function _throw(err) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "throw", err); } _next(undefined); }); }; }

var main =
/*#__PURE__*/
function () {
  var _ref = _asyncToGenerator(
  /*#__PURE__*/
  regeneratorRuntime.mark(function _callee() {
    var scalar, absTensor, range, filled, ones, zeros, ceiled, floored, spaced, eye, eye10;
    return regeneratorRuntime.wrap(function _callee$(_context) {
      while (1) {
        switch (_context.prev = _context.next) {
          case 0:
            scalar = tf.scalar(3.14);
            scalar.print();
            absTensor = tf.tensor1d([-10, -20, -30, -40, -50]).abs();
            absTensor.print();
            range = tf.range(0, 100, 10);
            range.print();
            filled = tf.fill([2, 3], 5);
            filled.print();
            ones = tf.ones([4, 4]);
            ones.print();
            zeros = tf.zeros([5, 5]);
            zeros.print();
            ceiled = tf.tensor1d([.6, 1.1, -3.3]).ceil();
            ceiled.print();
            floored = tf.tensor1d([.6, 1.1, -3.3]).floor();
            floored.print();
            spaced = tf.linspace(0, 1, 10);
            spaced.print();
            eye = tf.eye(5);
            eye.print();
            eye10 = tf.eye(10);
            eye10.print();

          case 22:
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
