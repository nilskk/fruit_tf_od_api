import tensorflow as tf


def read_tfrecord(tfrecord_path):
    record_list = []

    features = {
        # Extract features using the keys set during creation
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string)
    }

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    for record in dataset:
        sample = tf.io.parse_single_example(record, features)

        image = tf.io.decode_jpeg(sample["image/encoded"]).numpy()

        filename = sample['image/filename'].numpy().decode('UTF-8')

        record_dict = {'image': image, 'filename': filename}

        record_list.append(record_dict)

    return record_list


def get_steps_per_epoch(tfrecord_path,
                        batch_size):
    num_train_images = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
    steps_per_epoch = math.ceil(num_train_images / batch_size)

    return steps_per_epoch