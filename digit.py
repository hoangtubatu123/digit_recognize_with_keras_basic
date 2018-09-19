import pandas as  pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sub = pd.read_csv("data/sample_submission.csv")
all_X_train = train.drop(["label"], axis = 1)
all_y_train = train['label']
#normallize data
all_X_train = (all_X_train - np.mean(all_X_train)) / 256
test = (test - np.mean(test)) / 256
#split data train and test
X_train, X_test, y_train, y_test = train_test_split(
    all_X_train, all_y_train, test_size=0.15,random_state=0)

#revert to use CNN
X_train = X_train.values.reshape(X_train.shape[0], 28,28,1)
X_test = X_test.values.reshape(X_test.shape[0], 28,28,1)
test = test.values.reshape(test.shape[0], 28,28,1)
#one-hot coding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# setup and build model
model = Sequential()
#conv net
model.add(Conv2D(30, (5, 5), input_shape=(28,28,1) ,activation='relu', padding = "same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu', padding = "same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
# fully connected
model.add(Dense(128, activation = "relu"))
model.add(Dense(50, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

#complile model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#fit model
model.fit(X_train, y_train,
          batch_size=128, epochs=30,
          verbose=2,
          validation_data=(X_test, y_test))
#predict class 
predictions = model.predict_classes(test, batch_size=64)

image_id = sub["ImageId"]
submission_df = {"ImageId": image_id,
                 "Label": predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv("data/submission.csv",index=False)
