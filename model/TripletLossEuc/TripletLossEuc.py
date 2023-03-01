import os
import numpy as np
from tensorflow import reduce_sum, square, GradientTape, maximum
from tensorflow.keras import models, layers, optimizers, metrics, callbacks, utils
from tensorflow.data import Dataset

class TripletLossEuc:
    def __init__(self, type="flows"):
        self.type = type
        dim = 3 if type=="flows" else 1

        # Convolutional Neural Network
        self.cnn = models.Sequential()
        self.cnn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,dim)))
        self.cnn.add(layers.MaxPooling2D((3, 3)))

        self.cnn.add(layers.Conv2D(48, (3, 3), activation='relu'))
        self.cnn.add(layers.MaxPooling2D((2, 2)))

        self.cnn.add(layers.Conv2D(48, (3, 3), activation='relu'))
        self.cnn.add(layers.MaxPooling2D((2, 2)))

        self.cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.cnn.add(layers.MaxPooling2D((2, 2)))

        self.cnn.add(layers.GlobalAveragePooling2D())
        self.cnn.add(layers.Dense(48, activation='relu'))

        # -----------------------------------------------------------------------------------------------------
        
        class DistanceLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def call(self, anchor, positive, negative):
                ap_distance = reduce_sum(square(anchor - positive), -1)
                an_distance = reduce_sum(square(anchor - negative), -1)
                return (ap_distance, an_distance)


        anchor_input = layers.Input(name="anchor", shape=(224,224,dim))
        positive_input = layers.Input(name="positive", shape=(224,224,dim))
        negative_input = layers.Input(name="negative", shape=(224,224,dim))

        distances = DistanceLayer()(
            self.cnn(anchor_input),
            self.cnn(positive_input),
            self.cnn(negative_input),
        )

        siamese_network = models.Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )

        # -----------------------------------------------------------------------------------------------------

        class SiameseModel(models.Model):
            def __init__(self, siamese_network, margin=0.5):
                super(SiameseModel, self).__init__()
                self.siamese_network = siamese_network
                self.margin = margin
                self.loss_tracker = metrics.Mean(name="loss")

            def call(self, inputs):
                return self.siamese_network(inputs)

            def train_step(self, data):
                with GradientTape() as tape:
                    loss = self._compute_loss(data)

                gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

                self.optimizer.apply_gradients(
                    zip(gradients, self.siamese_network.trainable_weights)
                )

                self.loss_tracker.update_state(loss)
                return {"loss": self.loss_tracker.result()}

            def test_step(self, data):
                loss = self._compute_loss(data)

                self.loss_tracker.update_state(loss)
                return {"loss": self.loss_tracker.result()}

            def _compute_loss(self, data):
                ap_distance, an_distance = self.siamese_network(data)

                loss = ap_distance - an_distance
                loss = maximum(loss + self.margin, 0.0)
                return loss

            @property
            def metrics(self):
                return [self.loss_tracker]

        self.model = SiameseModel(siamese_network)
        self.model.compile(optimizer=optimizers.Adam(0.0001), weighted_metrics=[])

    def train(self, dbPath):
        path = fr"{dbPath}/{self.type}"

        # Get data from all the directories
        ds = [] #0->leftWrong same as leftPair, 1->rightWrong, 2->leftPair, 3->rightPair
        for label in ['wrong', 'good']:
            for side in ['left', 'right']:
                tmp = utils.image_dataset_from_directory(
                    fr"{path}/{label}/{side}",
                    color_mode="rgb" if self.type=="flows" else "grayscale",
                    image_size=(224,224),
                    batch_size=None,
                    labels=None,
                    shuffle=False
                )
                tmp = tmp.map(lambda x: x/255)
                ds.append(tmp)

        negative_input = ds[1]
        anchor_input = ds[2]
        positive_input = ds[3]

        train_dataset = Dataset.zip((anchor_input, positive_input, negative_input))

        size = len(os.listdir(fr"{path}/good/left"))
        train_size=int(size*0.8)

        val_dataset = train_dataset.skip(train_size)
        train_dataset = train_dataset.take(train_size)

        train_dataset = train_dataset.shuffle(1000).batch(64).prefetch(1)
        val_dataset = val_dataset.batch(64).prefetch(1)

        weightsPath = fr"model/TripletLossEuc/weight/{self.type}"
        if os.path.exists(weightsPath)==False:
            os.makedirs(weightsPath)
            
        checkPoint = callbacks.ModelCheckpoint(
            filepath=fr"{weightsPath}/weights",
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        self.model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[checkPoint])
        self.loadWeights(fr"{weightsPath}/weights")

    def loadWeights(self, weightsPath):
        self.model.load_weights(fr"{weightsPath}")

    def evaluate(self, dbPath):
        path = fr"{dbPath}/{self.type}"

        # Get data from all the directories
        ds = [] #0->leftWrong same as leftPair, 1->rightWrong, 2->leftPair, 3->rightPair
        for label in ['wrong', 'good']:
            for side in ['left', 'right']:
                tmp = utils.image_dataset_from_directory(
                    fr"{path}/{label}/{side}",
                    color_mode="rgb" if self.type=="flows" else "grayscale",
                    image_size=(224,224),
                    batch_size=None,
                    labels=None,
                    shuffle=False
                )
                tmp = tmp.map(lambda x: x/255)
                ds.append(tmp)

        negative_input = ds[1]
        anchor_input = ds[2]
        positive_input = ds[3]
        data = Dataset.zip((anchor_input, positive_input, negative_input))

        self.model.evaluate(data)

    def extractFeatures(self, dbPath):
        path = fr"{dbPath}/{self.type}"
        left = utils.image_dataset_from_directory(
            fr"{path}/good/left",
            color_mode="rgb" if self.type=="flows" else "grayscale",
            image_size=(224,224),
            batch_size=None,
            labels=None,
            shuffle=False
        )
        left = left.map(lambda x: x/255)

        right = utils.image_dataset_from_directory(
            fr"{path}/good/right",
            color_mode="rgb" if self.type=="flows" else "grayscale",
            image_size=(224,224),
            batch_size=None,
            labels=None,
            shuffle=False
        )
        right = right.map(lambda x: x/255)


        left = left.batch(1).prefetch(10)
        featuresLeft = self.cnn.predict(left)
        right = right.batch(1).prefetch(10)
        featuresRight = self.cnn.predict(right)

        return featuresLeft, featuresRight

    def computeSimilarity(self, v1, v2):
        return np.sqrt(np.sum(np.square(v1 - v2)))
