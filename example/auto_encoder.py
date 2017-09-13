import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf


def batch_generator(features, batch_size=50, n_epochs=1000):
    """
    Batch generator for the iris dataset
    """

    # Generate batches
    for epoch in range(n_epochs):
        start_index = 0
        while start_index != -1:
            # Calculate the end index of the batch to generate
            end_index = start_index + batch_size if start_index + batch_size < n else -1

            yield features[start_index:end_index]

            start_index = end_index


# Auto Encoder
class TF_AutoEncoder:
    def __init__(self, features, labels, dtype=tf.float32):
        self.features = features
        self.labels = labels
        self.dtype = dtype

        self.encoder = dict()

    def fit(self, n_dimensions):
        graph = tf.Graph()
        with graph.as_default():

            # Input variable
            X = tf.placeholder(self.dtype, shape=(None, self.features.shape[1]))

            # Network variables
            encoder_weights = tf.Variable(tf.random_normal(shape=(self.features.shape[1], n_dimensions)))
            encoder_bias = tf.Variable(tf.zeros(shape=[n_dimensions]))

            decoder_weights = tf.Variable(tf.random_normal(shape=(n_dimensions, self.features.shape[1])))
            decoder_bias = tf.Variable(tf.zeros(shape=[self.features.shape[1]]))

            # Encoder part
            encoding = tf.nn.sigmoid(tf.add(tf.matmul(X, encoder_weights), encoder_bias))

            # Decoder part
            predicted_x = tf.nn.sigmoid(tf.add(tf.matmul(encoding, decoder_weights), decoder_bias))

            # Define the cost function and optimizer to minimize squared error
            cost = tf.reduce_mean(tf.pow(tf.subtract(predicted_x, X), 2))
            optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session(graph=graph) as session:
            # Initialize global variables
            session.run(tf.global_variables_initializer())

            for batch_x in batch_generator(self.features):
                self.encoder['weights'], self.encoder['bias'], _ = session.run([encoder_weights, encoder_bias, optimizer],
                                                                            feed_dict={X: batch_x})

    def reduce(self):
        return np.add(np.matmul(self.features, self.encoder['weights']), self.encoder['bias'])

if __name__ == "__main__":
    # plt.close()
    color_mapping = {0: sns.xkcd_rgb['bright purple'], 1: sns.xkcd_rgb['lime'], 2: sns.xkcd_rgb['ochre']}

    IRIS_TRAINING = "iris_training.csv"
    IRIS_TEST = "iris_test.csv"
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    n = len(training_set.data)
    random_idx = np.random.permutation(n)
    features, labels = training_set.data[random_idx], training_set.target[random_idx]

    # Create an instance and encode
    tf_ae = TF_AutoEncoder(features, labels)

    tf_ae.fit(n_dimensions=2)

    auto_encoded = tf_ae.reduce()

    colors = list(map(lambda x: color_mapping[x], labels))

    plt.scatter(auto_encoded[:, 0], auto_encoded[:, 1], c=colors)

    plt.show()