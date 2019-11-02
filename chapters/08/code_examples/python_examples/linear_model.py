from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np 


def main():
  # get our data from sklearn's make_blobs method
  X, y = make_blobs(n_samples=1000, centers=2, random_state=50)

  # split our model into training and testing portions
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
  # create a sequential model, add a dense layer and train it
  model = Sequential()
  model.add(Dense(1, input_shape=(2,), activation='sigmoid'))
  model.compile(Adam(lr=0.05), loss='mse', metrics=['accuracy'])
  model.fit(X_train, y_train, epochs=100)
  
  # evaluate the model and print the results
  eval_result = model.evaluate(X_test, y_test)
  print('Loss: {}\nAccuracy: {}'.format(eval_result[0], eval_result[1]))

  # print the model summary
  model.summary()

  # save the model
  save_path = './output/linear_model'
  model.save(save_path)
  print('Model saved to: {}'.format(save_path))

  # make a sample prediction and print the results
  prediction = model.predict(np.array([[1, 2]]))
  print('Predicted result: {}'.format(prediction))

if __name__ == '__main__':
  main()