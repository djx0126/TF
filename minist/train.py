import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from minist import configs
from minist.data_model import DataModel
from minist.input_data import InputData

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")

FLAGS = flags.FLAGS



partial_for_train = 0.8

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

traindata = mnist.train

dataset = mnist
input_data = InputData(dataset)

train_data, test_data = input_data.random_pick(partial_for_train)

init_scale = 0.05
batch_size = 50

def get_config():
    if FLAGS.model == "small":
        print("use small config")
        return configs.SmallConfig()
    elif FLAGS.model == "medium":
        print("use medium config")
        return configs.MidConfig()
    elif FLAGS.model == "large":
        print("use large config")
        return configs.LargeConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

config = get_config()
eval_config = get_config()
eval_config.batch_size = 0 # means all data
eval_config.num_epochs = 1

# valid on test data
train_data = train_data
test_data = test_data
valid_data = test_data
    # train_data, valid = data.random_pick(0.7)
    # valid_data = valid

def run_epoch(session, model, eval_op=None, iter=0, verbose=False):
    fetches = {
        "accuracy": model.accuracy,
        "cost": model.cost
    }

    keep_prob = 1.0
    if eval_op is not None:
        fetches["eval_op"] = eval_op
        keep_prob = 0.5

    a_input_data = model.data.next() if model.is_training else model.data.data()

    feed_dict = {model.keep_prob: keep_prob, model.mixed_data: a_input_data}

    vals = session.run(fetches, feed_dict)

    if verbose:
        accuracy = vals["accuracy"]
        cost = vals["cost"]
        print("step %d, accuracy %g, current cost: %g" % (
            iter, accuracy, cost))

    return vals

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)

    print("setup models")
    with tf.name_scope("Train"):
        train_input = InputData(train_data.data(), batch_size=batch_size, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = DataModel(is_training=True, config=config, data_input=train_input)
        tf.summary.scalar("Training Loss", m.cost)

    with tf.name_scope("Valid"):
        valid_input = InputData(valid_data.data(), name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = DataModel(is_training=False, config=config, data_input=valid_input)
        tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
        test_input = InputData(test_data.data(), name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = DataModel(is_training=False, config=eval_config, data_input=test_input)

    print("start to run epoch")
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
        for i in range(config.num_epochs):

            verbose = True
            if i % 100 == 0:
                verbose = True
            else:
                verbose = False
            vals = run_epoch(session, m, eval_op=m.train_op, verbose=verbose, iter=i)

            if i % 1000 == 0:
                vals = run_epoch(session, mvalid)
                print("Epoch: %d Valid accuracy: %f, cost:%.3f" % (
                    i, vals["accuracy"], vals["cost"]))

            if i % 10000 == 0:
                vals = run_epoch(session, mtest)
                print("Test accuracy: %f" % vals["accuracy"])

        vals = run_epoch(session, mtest)
        print("Test accuracy: %f" % vals["accuracy"])

        if FLAGS.save_path:
            print("Saving model to %s." % FLAGS.save_path)
            sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

print("end")
