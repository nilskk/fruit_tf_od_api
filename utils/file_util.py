import tf


def read_tfrecord(tfrecord_path):
    image_list = []
    filename_list = []

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

        image_list.append(image)
        filename_list.append(filename)

    record_dict = {'image': image_list, 'filename': filename_list}

    return record_dict