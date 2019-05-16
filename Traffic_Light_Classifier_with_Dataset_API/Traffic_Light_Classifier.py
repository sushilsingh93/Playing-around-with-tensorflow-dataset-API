import tensorflow as tf
import os
import numpy as np
from datagen_traffic_light import ImageDataGenerator

dataset_path = "D:\Data\Traffic_light_images"
modelpath = "D:\Data\Traffic_light_images\Model\ "

batch_size = 100
number_class = 3

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

imagepath, labels = dataset_creation(dataset_path)

paths = imagepath
labels = labels
permutation = np.random.permutation(len(labels))
shuffled_image_paths = []
shuffled_labels = []
for i in permutation:
    shuffled_image_paths.append(paths[i])
    shuffled_labels.append(labels[i])

# *********************** Dataset Split **********************
# Training 90% of training data
train_image_paths = shuffled_image_paths[: int(len(shuffled_image_paths) * 0.9)]
train_labels = shuffled_labels[: int(len(shuffled_labels) * 0.9)]
# Validation 90% of training data
valid_image_paths = shuffled_image_paths[int(len(shuffled_image_paths) * 0.9): ]
valid_labels = shuffled_labels[int(len(shuffled_labels) * 0.9): ]


with tf.device('/cpu:0'):
    # Data loading and preprocessing on the cpu for Training data
    tr_data = ImageDataGenerator(train_image_paths, train_labels,number_class,  mode='training', batch_size=batch_size, shuffle=True)
    valid_data = ImageDataGenerator(valid_image_paths, valid_labels, number_class, mode='validation', batch_size=batch_size, shuffle=False)

    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
    next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(valid_data.data)

def simple_nn(in_data, is_training, reuse):
    with tf.variable_scope('SIMPLE_NN', reuse=reuse):
        flat_nn = tf.layers.flatten(in_data)
        bn = tf.layers.batch_normalization(flat_nn)
        fc_1 = tf.layers.dense(bn, 50)
        fc_2 = tf.layers.dense(fc_1, 50)
        fc_2 = tf.layers.dropout(fc_2, training= is_training) # default rate is 0.5
        fc_3 = tf.layers.dense(fc_2, 3)
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

