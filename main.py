import tensorflow as tf
from tensorflow.keras import layers


class VGGBlock(layers.Layer):
    """VGG model building block
    if last_layer of the block:
        conv -> relu -> batchnorm -> pooling
    else:
        conv -> relu -> batchnorm -> dropout
    """
    def __init__(self, name=None, num_filter=32,
                 dropout_rate=0.3, is_last=False, **kwargs):
        super(VGGBlock, self).__init__(name=name, **kwargs)
        self.conv = layers.Conv2D(filters=num_filter, kernel_size=3, padding="same")
        self.relu = layers.ReLU()
        self.batchnorm = layers.BatchNormalization(epsilon=1e-5)
        self.dropout = layers.Dropout(dropout_rate)
        self.pooling = layers.MaxPool2D(pool_size=(2, 2))
        self.is_last = is_last

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.relu(x)
        x = self.batchnorm(x)
        if self.is_last:
            x = self.pooling(x)
        else:
            x = self.dropout(x)
        return x


class VGGBuilder(tf.keras.Model):
    """VGG model builder"""
    def __init__(self, name="vgg16", **kwargs):
        super(VGGBuilder, self).__init__(name=name, **kwargs)
        if name == "vgg16":
            # block num, filter num, dropout rate
            self.structures = {"stage1": [2, 64, 0.3],
                               "stage2": [2, 128, 0.4],
                               "stage3": [3, 256, 0.4],
                               "stage4": [3, 512, 0.4],
                               "stage5": [3, 512, 0.4]}
            self.blocks = []
            for key in self.structures:
                num_block, num_filter, dropout_rate = self.structures[key]
                for idx, block in enumerate(range(num_block), start=1):
                    is_last = (num_block == idx)
                    self.blocks.append(VGGBlock(f"{key}_{idx}", num_filter,
                                                dropout_rate, is_last))
        self.dropout = layers.Dropout(0.5)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(512)
        self.relu = layers.ReLU()
        self.batchnorm = layers.BatchNormalization(epsilon=1e-5)

    def call(self, inputs):
        x = inputs
        for layer in self.blocks:
            x = layer(x)
            print(x.shape)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        print(x.shape)
        return x


class VGGClassifier(tf.keras.Model):
    """Classifier that outputs softmax probabilities"""
    def __init__(self, name="vgg16_classifier",
                 input_shapes=(32, 32, 3), num_classes=10, **kwargs):
        super(VGGClassifier, self).__init__(name=name, **kwargs)
        self.input_layer = layers.InputLayer(input_shape=input_shapes)
        self.vggbuilder = VGGBuilder()
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(num_classes)
        self.softmax = layers.Softmax()

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.vggbuilder(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.softmax(x)
        print(x.shape)
        return x


def load_dataset():
    MEAN_IMAGE = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
    STD_IMAGE = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    trainset = trainset.map(
        lambda image, label: (
            (tf.cast(image, tf.float32) / 255.0) - MEAN_IMAGE / STD_IMAGE,
            tf.cast(label, tf.float32))
    ).shuffle(buffer_size=1024).repeat().batch(128)

    testset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    testset = testset.map(
        lambda image, label: (
            (tf.cast(image, tf.float32) / 255.0) - MEAN_IMAGE / STD_IMAGE,
            tf.cast(label, tf.float32))
    ).batch(128)

    return trainset, testset


vgg16 = VGGClassifier(num_classes=10 + 1)

trainset, testset = load_dataset()
for image, label in trainset.take(1):
    vgg16(image)
