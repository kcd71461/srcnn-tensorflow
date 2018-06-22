import tensorflow as tf

config = tf.contrib.training.HParams(
    image_size=33,
    label_size=21,
    scale=3,
    stride=21,
    learning_rate=0.0001,
    batch_size=64,
    log_path="./logs",
    result_dir="./result"
)
