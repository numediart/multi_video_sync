import os
from tensorflow.keras import models, layers, callbacks
from keras.utils import np_utils
import numpy as np

class DenseDelay:
    def __init__(self, type="flows"):
        self.type = type
        self.model = models.Sequential([
            layers.Dense(64, input_dim = 400, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(40, activation="softmax")
        ])

        self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    def train(self, pathSimilarityVectorsArray: str):
        trainDS = np.load(fr"{pathSimilarityVectorsArray}", allow_pickle=True)

        X = trainDS[:,0].tolist()
        y = trainDS[:,3].tolist()

        X = np.array(X)
        y = np_utils.to_categorical(y, num_classes=40)

        size = int(len(X)*0.8)

        X_train, X_val = (X[:size], X[size:])
        y_train, y_val = (y[:size], y[size:])

        weightsPath = fr"model/DenseDelay/weight/{self.type}"
        if os.path.exists(weightsPath)==False:
            os.makedirs(weightsPath)

        checkPoint = callbacks.ModelCheckpoint(
            filepath=fr"{weightsPath}/weights",
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        self.model.fit(X_train, y_train, validation_data=[X_val, y_val], callbacks=[checkPoint], epochs=50)
        self.model.load_weights(fr"{weightsPath}/weights")

    def loadWeights(self, weightsPath):
        self.model.load_weights(fr"{weightsPath}/weights").expect_partial()

    def evaluate(self, pathSimilarityVectorsArray):
        print(fr"{pathSimilarityVectorsArray}")
        db = np.load(fr"{pathSimilarityVectorsArray}", allow_pickle=True)

        X = db[:,0].tolist()
        y = db[:,3].tolist()

        X = np.array(X)
        y = np_utils.to_categorical(y, num_classes=40)

        self.model.evaluate(X, y)