import tensorflow as tf

class DataModel(object):
    def __init__(self, is_training, config, data_input):
        self._is_training = is_training
        self._config = config
        self._data_input = data_input

        data_width = data_input.data().shape[1]

        self._mixed_data = tf.placeholder(tf.float32, [None, data_width], name="input_data")
        self._keep_prob = tf.placeholder(tf.float32)
        y_ = tf.slice(self._mixed_data, [0, 0], [-1, data_input.label_size])
        x = tf.slice(self._mixed_data, [0, data_input.label_size], [-1, -1])

        x_width = data_width - data_input.label_size

        nc_1_size = self._config.fc_size
        with tf.name_scope("fc1"):
            W_fc1 = self.weight_variable([x_width, nc_1_size], "fc1")
            tf.add_to_collection('losses', tf.nn.l2_loss(W_fc1))
            b_fc1 = self.bias_variable([nc_1_size], "fc1")
            h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, self._keep_prob)

        with tf.name_scope("fc2"):
            W_fc2 = self.weight_variable([nc_1_size, data_input.label_size], "fc2")
            tf.add_to_collection('losses', tf.nn.l2_loss(W_fc2))
            b_fc2 = self.bias_variable([data_input.label_size], "fc2")
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                    logits=y_conv)

            cost = tf.reduce_mean(cross_entropy)
            # total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            # cost = tf.reduce_mean(tf.pow(y_conv - y_, 2) + config.beta * total_loss)

        self._cost = cost

        with tf.name_scope('adam_optimizer'):
            self._train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self._accuracy = tf.reduce_mean(correct_prediction)

        # self._cost = tf.reduce_mean(tf.pow(y_conv - y_, 2) + config.beta * total_loss)

    def weight_variable(self, shape, scope_name):
        with tf.variable_scope(scope_name):
            w = tf.get_variable("w", shape, dtype=tf.float32)
            # print("variable w name = " + w.name + ", reuse?" + str(tf.get_variable_scope().reuse))
            return w

    def bias_variable(self, shape, scope_name):
        with tf.variable_scope(scope_name):
            return tf.get_variable("b", shape, dtype=tf.float32)

    def conv2d(self, x, W):
        padding = 'VALID'
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    def max_pool_2x2(self, x):
        padding = 'VALID'
        return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                              strides=[1, 2, 1, 1], padding=padding)

    @property
    def is_training(self):
        return self._is_training

    @property
    def data(self):
        return self._data_input

    @property
    def mixed_data(self):
        return self._mixed_data

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def train_op(self):
        return self._train_op

    @property
    def cost(self):
        return self._cost

    @property
    def accuracy(self):
        return self._accuracy