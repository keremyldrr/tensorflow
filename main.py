import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
import os
from TFRecorder import TFRecorder

# records
train_record = TFRecorder('images/resimler', 'traindata.tfrecords')

val_record = TFRecorder('images/resimler', 'valdata.tfrecords')
# Learning params
learning_rate = 0.001
num_epochs = 10
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = len(train_record.classes)
train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "finetune_alexnet/test_objects"
checkpoint_path = "finetune_alexnet/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

x = tf.placeholder(tf.float32, shape=[batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_images, tr_labels = train_record.read_decode_Record(batchsize=batch_size)
val_images, val_labels = val_record.read_decode_Record(batchsize=batch_size)

train_batches_per_epoch = train_record.count // batch_size
val_batches_per_epoch = val_record.count // batch_size
with tf.Session() as sess:
    # Initialize all variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    saver.restore(sess,'finetune_alexnet/model_epoch9.ckpt')
    # Load the pretrained weights into the non-trainable layer
    # model.load_initial_weights(sess)
    #
    # print("{} Start training...".format(datetime.now()))
    # print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
    #                                                   filewriter_path))

    # Loop over number of epochs
    # for epoch in range(num_epochs):
    #
    #     print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
    #
    #     step = 1
    #     while step < train_batches_per_epoch:
    #         # Get a batch of images and labels
    #
    #         batch_xs, batch_ys = sess.run([train_images, tr_labels])
    #         print(step,batch_xs.shape[0])
    #
    #         if batch_xs.shape[0] < 128:
    #             continue
    #         batch_ys = sess.run(tf.one_hot(batch_ys, num_classes))
    #
    #         # And run the training op
    #         sess.run(train_op, feed_dict={x: batch_xs,
    #                                       y: batch_ys,
    #                                       keep_prob: dropout_rate})
    #
    #         # Generate summary with the current batch of data and write to file
    #         if step % display_step == 0:
    #             s = sess.run(merged_summary, feed_dict={x: batch_xs,
    #                                                     y: batch_ys,
    #                                                     keep_prob: 1.})
    #             writer.add_summary(s, epoch * train_batches_per_epoch + step)
    #     #
    #         step += 1
    #
    # print("{} Saving checkpoint of model...".format(datetime.now()))
    #
    # # save checkpoint of the model
    # checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch) + '.ckpt')
    # save_path = saver.save(sess, checkpoint_name)
    #
    # print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

    print("{} Start validation".format(datetime.now()))
    test_acc = 0.
    test_count = 0

    for epoch in range(num_epochs):
        for step in range(val_batches_per_epoch):  #

            batch_tx, batch_ty = sess.run([val_images, val_labels])
            print(step,batch_tx.shape[0])
            if batch_tx.shape[0] < 128:
                continue
            acc = sess.run(accuracy, feed_dict={x: batch_tx,
                                                y: sess.run(tf.one_hot(batch_ty, num_classes)),
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
            test_acc = float(test_acc/test_count)
            print(acc)
        print("Validation Accuracy =" + str(test_acc))

coord.request_stop()
coord.join(threads)
