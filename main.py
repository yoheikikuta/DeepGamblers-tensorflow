import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from tensorflow.keras import layers

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'classes',
    default=10,
    help="The number of classes of the classification.")

flags.DEFINE_float(
    'o',
    default=2.2,
    help="Weight for abstention class: (1 / o) abstention_prob.")


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
        self.conv = layers.Conv2D(filters=num_filter, kernel_size=3, padding="same",
                                  kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.relu = layers.ReLU()
        self.batchnorm = layers.BatchNormalization(epsilon=1e-5)
        self.dropout = layers.Dropout(dropout_rate)
        self.pooling = layers.MaxPool2D(pool_size=(2, 2))
        self.is_last = is_last

    def call(self, inputs, training=True):
        x = self.conv(inputs)
        x = self.relu(x)
        x = self.batchnorm(x, training)
        if self.is_last:
            x = self.pooling(x)
        else:
            x = self.dropout(x, training)
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
        self.dense = layers.Dense(512,
                                  kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.relu = layers.ReLU()
        self.batchnorm = layers.BatchNormalization(epsilon=1e-5)

    def call(self, inputs, training=True):
        x = inputs
        for layer in self.blocks:
            x = layer(x, training)
        x = self.dropout(x, training)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.batchnorm(x, training)
        return x


class VGGClassifier(tf.keras.Model):
    """Classifier that outputs softmax probabilities"""
    def __init__(self, name="vgg16_classifier",
                 input_shapes=(32, 32, 3), num_classes=10, **kwargs):
        super(VGGClassifier, self).__init__(name=name, **kwargs)
        self.input_layer = layers.InputLayer(input_shape=input_shapes)
        self.vgg = VGGBuilder()
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(num_classes,
                                  kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.softmax = layers.Softmax()

    def call(self, inputs, training=True):
        x = self.input_layer(inputs)
        x = self.vgg(x, training)
        x = self.dropout(x, training)
        x = self.dense(x)
        x = self.softmax(x)
        return x


# Gambler loss that proposed in the paper
def gambler_loss(model, x, y, o):
    # \sum_{i=1}^{m} y_i log(p_i + (1 / o) p_{m+1})
    EPS = 1e-5
    prob = model(x)
    prob = tf.clip_by_value(prob, EPS, 1.0 - EPS)
    class_pred, abstention = tf.split(prob, [prob.shape[1] - 1, 1], 1)
    abstention /= o
    weighted_prob = tf.concat([class_pred, abstention], 1)

    label_shape = y.shape
    extended_label = tf.concat([y, tf.constant(1.0, shape=[label_shape[0], 1])], 1)

    log_arg = tf.reduce_sum(extended_label * weighted_prob, 1)
    cross_ent = -tf.reduce_mean(tf.math.log(log_arg))

    return cross_ent


def train(model, optimizer, trainset, o):
    for step, (x_batch_train, y_batch_train) in enumerate(trainset):
        with tf.GradientTape() as tape:
            loss = gambler_loss(model, x_batch_train, y_batch_train, o)
            print(loss)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


def evaluate(model, testset):
    training = False
    predictions = np.array([], dtype=np.float32).reshape(0, 11)
    answers = np.array([], dtype=np.int8).reshape(0)
    for (x_batch_test, y_batch_test) in testset:
        preds = model(x_batch_test, training)
        predictions = np.vstack([predictions, preds.numpy()])
        answers = np.hstack([answers, y_batch_test.numpy().flatten()])

    return predictions, answers


def data_augmentation(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, tf.constant([[2, 2], [2, 2], [0, 0]]), "REFLECT")
    x = tf.image.random_crop(x, size=[32, 32, 3])
    return x


def load_dataset():
    MEAN = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
    STD = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # x_train, y_train = x_train[:128 * 10], y_train[:128 * 10]
    # x_test, y_test = x_test[:128 * 1], y_test[:128 * 1]
    trainset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    trainset = trainset.map(
        lambda image, label: (
            data_augmentation((tf.cast(image, tf.float32) / 255.0) - MEAN / STD),
            tf.squeeze(tf.cast(tf.one_hot(label, depth=FLAGS.classes), tf.float32)))
    ).shuffle(buffer_size=1024).batch(128)

    testset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    testset = testset.map(
        lambda image, label: (
            (tf.cast(image, tf.float32) / 255.0) - MEAN / STD,
            label)
    ).batch(128)

    return trainset, testset


def main(argv):
    trainset, testset = load_dataset()
    vgg16 = VGGClassifier(num_classes=FLAGS.classes + 1)  # +1 is for abstention class
    optimizer = tf.keras.optimizers.SGD(learning_rate=5e-3, momentum=0.9)

    for epoch in range(2):
        print(f"Start of epoch {epoch + 1}")
        train(vgg16, optimizer, trainset, FLAGS.o)

    predictions, answers = evaluate(vgg16, testset)
    print(sum([np.argmax(elem) for elem in predictions] == answers))


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(main)
