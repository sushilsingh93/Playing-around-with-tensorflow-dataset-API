
import tensorflow as tf
import os
import numpy as np
from datagenerator import ImageDataGenerator

dataset_path_train = "D:\Data\MNIST\MNIST Dataset JPG format\MNIST-JPG-training"
dataset_path_test = "D:\Data\MNIST\MNIST Dataset JPG format\MNIST-JPG-testing"
modelpath = "D:\Data\MNIST\MNIST Dataset JPG format\Model\ "
batch_size = 100
number_class = 10

def dataset_creation(dataset_path):
    imagepaths, labels = list(), list()
    classes = sorted(os.walk(dataset_path).__next__()[1])  # List the directory
    label_int = 0
    for c in classes:
        class_dir = os.path.join(dataset_path, c)
        for sample_img in os.walk(class_dir).__next__()[2]:
            if sample_img.endswith('.jpg') or sample_img.endswith('.jpeg'):
                image_path = os.path.join(class_dir, sample_img)
                imagepaths.append(image_path)
                labels.append(label_int)
        label_int += 1
    return imagepaths, labels

imagepaths_traindata, labels_traindata = dataset_creation(dataset_path_train)
imagepaths_testdata, labels_testdata = dataset_creation(dataset_path_test)

paths = imagepaths_traindata
labels = labels_traindata
permutation = np.random.permutation(len(labels))
shuffled_image_paths = []
shuffled_labels = []
for i in permutation:
    shuffled_image_paths.append(paths[i])
    shuffled_labels.append(labels[i])

# *********************** Dataset Split **********************
# Training 80% of training data
train_image_paths = shuffled_image_paths[: int(len(shuffled_image_paths) * 0.8)]
train_labels = shuffled_labels[: int(len(shuffled_labels) * 0.8)]
# Validation 20% of training data
valid_image_paths = shuffled_image_paths[int(len(shuffled_image_paths) * 0.8): ]
valid_labels = shuffled_labels[int(len(shuffled_labels) * 0.8): ]

with tf.device('/cpu:0'):
    # Data loading and preprocessing on the cpu for Training data
    tr_data = ImageDataGenerator(train_image_paths, train_labels,number_class,  mode='training', batch_size=batch_size, shuffle=True)
    valid_data = ImageDataGenerator(valid_image_paths, valid_labels, number_class, mode='validation', batch_size=batch_size, shuffle=False)
    test_data = ImageDataGenerator(imagepaths_testdata, labels_testdata, number_class, mode='test', batch_size=batch_size, shuffle=False)

    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
    next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(valid_data.data)
# test_init_op = test_data.data.make_one_shot_iterator()
test_init_op = iterator.make_initializer(test_data.data)

def simple_nn(in_data, is_training, reuse):
    with tf.variable_scope('SIMPLE_NN', reuse=reuse):
        flat_nn = tf.layers.flatten(in_data)
        bn = tf.layers.batch_normalization(flat_nn)
        fc_1 = tf.layers.dense(bn, 256)
        fc_2 = tf.layers.dense(fc_1, 256)
        fc_2 = tf.layers.dropout(fc_2, training= is_training) # default rate is 0.5
        fc_3 = tf.layers.dense(fc_2, 10)
        fc_3 = tf.nn.softmax(fc_3) if not is_training else fc_3
    return fc_3

nn_prediction_logits_train = simple_nn(next_batch[0], is_training=True, reuse= False)
nn_prediction_logits_valid = simple_nn(next_batch[0], is_training=False, reuse= True)

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=next_batch[1], logits=nn_prediction_logits_train))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

equality = tf.equal(tf.arg_max(nn_prediction_logits_valid, 1), tf.arg_max(next_batch[1], 1))
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
saver = tf.train.Saver()


epochs = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(training_init_op)
    itr_cycle = int(len(train_image_paths) / batch_size)
    for j in range(epochs):
        print("Epoch: {}".format(j+1))
        for i in range(itr_cycle):
            l, _, acc = sess.run([loss, optimizer, accuracy])
            if i%50 == 0:
                print("Itteration_count: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc*100))

        # now setup the validation run
        valid_iters = 100
        # re-initialize the iterator, but this time with validation data
        sess.run(validation_init_op)
        avg_acc = 0
        for i in range(valid_iters):
            acc = sess.run([accuracy])
            avg_acc += acc[0]
        print("Average validation set accuracy over {} iterations is {:.2f}%".format(valid_iters,
                                                                                     (avg_acc / valid_iters) * 100))
        j += 1

    save_path = saver.save(sess, modelpath)
    print("Model saved in file: %s" % save_path)

    sess.run(test_init_op)
    test_accuracy = 0
    try:
        while True:
            acc = sess.run([accuracy])
            test_accuracy += acc[0]
    except tf.errors.OutOfRangeError:
        print("OUT OF RANGE VALUES")
        pass
    print("\nTest accuracy: {:.3f}".format(test_accuracy/(len(labels_testdata)/batch_size) * 100))


# ********************************************************************** #
# Testing when model file is available with us
# ********************************************************************** #
import matplotlib.image as mpimage
import matplotlib.pyplot as plt

def parse_func(img_path):
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_png(img_string, channels=1)
    img_resized = tf.image.resize_images(img_decoded, [28, 28])
    return img_resized

img = mpimage.imread(imagepaths_testdata[0])
plt.imshow(img)

img_path = [imagepaths_testdata[0]]
img_tensor = tf.convert_to_tensor(img_path, dtype=tf.string)
data = tf.data.Dataset.from_tensor_slices(img_tensor)
data = data.map(parse_func)
data = data.batch(1)

iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
next_data = iterator.get_next()
test_init_op = iterator.make_initializer(data)

model_prediction = simple_nn(next_data, is_training=False, reuse= True)
_equality = tf.equal(tf.arg_max(model_prediction, 1), tf.constant([labels_testdata[0]], dtype=tf.int64))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(test_init_op)
    load_path = saver.restore(sess, modelpath)
    print("Model restored from file: %s" % modelpath)

    logit_model, eql = sess.run([model_prediction, _equality])

    if eql == True:
        print("MATCHED!!")