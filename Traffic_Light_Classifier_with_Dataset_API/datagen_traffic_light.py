import numpy as np
import tensorflow as tf



class ImageDataGenerator(object):

    def __init__(self, image_paths, labels, n_class, mode, batch_size, shuffle=True, buffer_size=1000):
        self.image_paths = image_paths
        self.labels = labels
        self.number_class = n_class
        self.data_size = len(self.labels)

        if shuffle:
            self._shuffle_lists()

        # convert data lists to TF tensor
        self.image_paths = tf.convert_to_tensor(self.image_paths, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)

        # dataset object
        data = tf.data.Dataset.from_tensor_slices((self.image_paths, self.labels))

        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)
        if mode == 'validation' or mode == 'test':
            data = data.map(self._parse_function_valid, num_parallel_calls=8)

        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        data = data.batch(batch_size=batch_size)
        if mode =='training' or mode == 'validation':
            data = data.repeat()

        self.data = data

    def _shuffle_lists(self):
        paths = self.image_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)

        self.image_paths = []
        self.labels = []
        for i in permutation:
            self.image_paths.append(paths[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, img_path, label):
        one_hot = tf.one_hot(label, self.number_class)

        img_string = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [32, 32])

        return img_resized, one_hot

    def _parse_function_valid(self, img_path, label):
        one_hot = tf.one_hot(label, self.number_class)

        img_string = tf.read_file(img_path)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [32, 32])

        return img_resized, one_hot