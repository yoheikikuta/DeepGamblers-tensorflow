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
    def __init__(self, name="vgg16", input_shapes=(32, 32, 3), **kwargs):
        super(VGGBuilder, self).__init__(name=name, **kwargs)
        self.input_layer = layers.InputLayer(input_shape=input_shapes)
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
        x = self.input_layer(inputs)
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


(x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))

vgg16 = VGGBuilder()

for image, label in train_dataset.take(1):
    image = tf.dtypes.cast(image, tf.float32)
    vgg16(tf.expand_dims(image, 0))
