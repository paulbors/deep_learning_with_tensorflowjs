# Chapter 7 - Importing and Running Existing Models in TensorFlow.js Applications

In the last chapters we took at the workings of the TensorFlow.js Core API and became familiar with how to build models from scratch using the Layers API. In this chapter we will learn how to work with existing models that were created outside of the TensorFlow.js environment. After completing this chapter you should learn the following:

- How to save TensorFlow.js models for later use
- How to import existing models created using the Python version of TensorFlow or Keras
- How to convert saved TensorFlow.js models to Keras format
- How to retrain an imported model with new data

In this chapter we will be building a simple recommendation engine. Rather than building the recommendation engine ourselves we will import a pre-trained model. We will then accept user feedback to add new inputs to and retrain our model. This is very similar to how we would use TensorFlow.js in production.

## TensorFlow Deployment Strategy

As we learned in previous chapters, TensorFlow.js provides the ability to create and train models in the browser. This is a very powerful feature of the library and all of the models we’ve created so far were developed either in Node.js or the browser. However, training models can be both resource and time intensive. Developing deep learning models in the browser may not be the approach that we would want to take when building productions systems. Instead, we would want to develop our models outside of the browser using either TensorFlow (Python) or Keras (Python) or TFJS (JavaScript) like we did in previous chapters in this book. This way we can leverage the power of the user’s machine for development and that of a production server for retraining. Since most machine learning models must load a lot of data and will have to perform complex calculations on this data it makes sense to keep the development of our models offline and deploy the model to the browser for consumption.

## Saving Models

Before we talk about importing existing models into the browser we will briefly cover how to save models that we have developed using TensorFlow.js. In the previous chapter we saw how we were able to save the image recognition model we created in the previous chapter using the `model.save` method. Using this method we were able to save the model that we trained to a directory that lives on our computer. This works well if we are running a program on the server using Node.js; however, when performing work in the browser we do not have access to the user's file system. However, we do have alternative ways to save our models for future use.

When saving a model we use `model.save` we pass a string to the method as a parameter. This string is URL-like in nature and takes the following format: "<SCHEME>://<PATH>", where the scheme specifies where the model should be saved (i.e. to Local Storage or the file system in our example from chapter 6) and the path represents where the model should be saved. The call to this method is asynchronous so we treat like any other async method or function that returns a promise. When unwrapping the promise returned by this method call we receive a JSON object that contains information about the model including the model's weights and topology. We can call `model.save` on any `tf.Model` instance that

### Local Storage

It is possible to save our models directly to the browser by leveraging the Local Storage API. This API allows us to save data in the user's browser. Data stored here is read-only and does not expire and is saved across multiple browser sessions. This is useful if we are looking to retrain a model and save the retrained result to the user's browser. The scheme will be "localstorage://" and the path will be the name of our model, as demonstrated in the example below. We can then use the Local Storage API to retrieve and run our model.

```JavaScript
const savedModel = await model.save('localstoreage://model-to-save');
```

### IndexedDB

In addition to Local Storage, we have the option of saving our models to IndexedDB. IndexDB is a low-level API that is used for client-storage of data. The type of data that we can save to IndexedDB includes files and blobs (binary data). It is similar to Web Storage but was designed to enable high-performance searched of data stored to it and was designed to specifically store large amounts of data. If we are saving larger models it makes sense to save them to IndexedDB rather than Local Storage to take advantage of the storage efficiency and larger sized limit. Like saving the Local Storage the path will simply be the name of the file we wish to save. We can then use the IndexedDB API to retrieve and run our model.

```JavaScript
const savedModel = await model.save('indexeddb://model-to-save');
```

If we want to see what models we have saved to IndexedDB (as well as Local Storage) we leverage the `tf.io` API.

```javascript
const modelList = await tf.io.listModels();
console.log(modelList);
```

We can also copy our models between Local Storage and IndexedDB using `tf.io.copyModel`.

```javascript
tf.io.copyModel("localstorage://saved-model", "indexeddb://saved-model");
```

If wish to move a model rather than copy it from one location to another we can use `tf.io.moveModel`.

```JavaScript
tf.io.moveModel('localstorage://saved-model', 'indexeddb://saved-model');
```

Last, if we wanted to remove a model from either IndexedDB or Local Storage we can pass the scheme and path of the model we want to delete to `tf.io.removeModel`.

```javascript
tf.io.removeModel("indexeddb://saved-model");
```

### Downloading to the User's Desktop

Though we cannot save our models directly to the user's desktop we can initiate a download of the model as a file download. This will trigger a download of the model that will create a user experience similar to downloading a PDF or other file from the browser. The download result will be a JSON file named after the path that was specified to initialize the download (i.e. `model-to-save.json`).

```JavaScript
const savedModel = await model.save('download://model-to-save');
```

The resulting file will contain the model weights along with its topology. Along with this file comes a binary file that contains the weights of our model. This file is typically named as such: `model-to-save.weights.bin`.

### Server-side Saving

Saving newly created or retrained models in the user's browser is a useful feature but what if we wanted to save the model offline on our server somewhere? The good news is that TensorFlow.js allows us to save our models to a remote server using HTTP requests. This means to save our model we can do so via POST request. In order to gain more control over the request that is being sent we can use the `tf.io.browserHTTPRequest` API that allows us to specify the HTTP method and headers of our request.

```JavaScript
await model.save(tf.io-browserHTTPRequest('http://yourserver.com/upload', {method: 'PUT', headers: {'header_key': 'header_value'}}));
```

The body of this POST request will be in the `multipart/form-data` format, which is a common MIME format used for uploading files to servers. The body of the request will contain `model.json` and `model.weights.bin` files. These files are formatted identically to the files generated by the `downloads://` scheme.

### Saving to the User's File System

This is the strategy we used in the previous chapter and is available in tfjs-node. Since NOde.js can access the user's file system directly we are abel to use the node bindings to save our `tf.Model` instance to the user's computer. The only difference is that we must obviously be running our program that will be saving our model in a Node.js environment after importing `@tensorflow/tfjs-node`. Once this is done we can call `model.save` using the file system scheme and specifying a path to save our file to.

```JavaScript
import * as tf from '@tensorflow/tfjs-node';

await model.save('file:///trained-models/model-to-save');
```

This command will create a `model.json` and `weights.bin` files in a directory called `/trained-models/model-to-save`. These files will be of the same format as the files created when using the `download` and `http` schemes.

### Converting TensorFlow.js tf.Models into Keras Format

Once we have saved a `tf.Model` instance via the `downloads://` or `file://` schemas we can convert from TensorFlow.js format to in the format that Keras uses to load models (HDF5). To do this we will have to use teh tensorflowjs_converter that ships with the the `tensorflowjs` Python package. Once installed into a Python environment we can convert our file to HDF5 format using the following bash command.

```bash
tensorflowjs_converter --input_format tensorflowjs --output_format keras ./saved-model.json /trained_models/saved-model.h5
```

Once converted to this format we can load it into a Python program via the Keras library using its API methods. Keras is out of the scope of this book so it is suggested to read up on the Keras documentation to learn how to import external models for use with the library.

## Importing a Keras Model into TensorFlow.js

TBD

## Importing a Python TensorFlow Model into TensorFlow.js

TBD

## Retraining Existing Models with User Input

TBD - code along tutorial pending changes to code examples

## Summary

In this chapter we learned how to import existing models for use in our browser-based applications. We also learned how to use these imported models to perform inference on user inputs that were fed into a simple recommendation engine. Next we reviewed the steps required to load models developed in Keras or the Python and C++ TensorFlow libraries into the browser using the `tf.loadModel` method. Last we discovered how we can refrain the models we imported into the browser with new data points collected from the user. In the next chapter we will focus on deployment and how e can write our code more efficiently for production use.
