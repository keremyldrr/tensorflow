import os
import tensorflow as tf
import cv2
import glob
from random import shuffle
import numpy as np
import sys


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def countData(recordname):
    size = 0
    for record in tf.python_io.tf_record_iterator(recordname):
        size += 1
    return size


class TFRecorder(object):

    def __init__(self, dataLoc, filepath, name='READER',shuffle_data=True):

        locs = os.listdir(dataLoc)
        locs.sort()
        self.classes = locs
        self.datapath = dataLoc
        if name == 'READER':
            self.count = countData(filepath)
            self.filepath = filepath
        elif name == 'WRITER':

            count = 0
            addrs = []
            labels = []
            for loc in self.classes:

                for a in glob.glob(self.datapath + '/' + loc + '/*'):

                    img = cv2.imread(a)
                    t = str(img)
                    if not t == 'None':
                        img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype(np.float32)
                        label = self.classes.index(loc)
                        labels.append(label)
                        addrs.append(img)
                        count += 1

            train_addrs=addrs[0:int(0.8*len(addrs))]
            train_labels=labels[0:int(0.8*len(addrs))]
            val_addrs=addrs[int(0.8*len(addrs)):len(addrs)]
            val_labels=labels[int(0.8*len(addrs)):len(addrs)]
            print(len(train_labels),len(val_labels))
            self.writeRecord(train_addrs,train_labels,'traindata.tfrecords')
            self.writeRecord(val_addrs,val_labels, 'valdata.tfrecords')
            self.count = count
            if shuffle_data:
                c = list(zip(addrs, labels))
                shuffle(c)
                addrs, labels = zip(*c)

            # Divide the hata into 80% train, 20% validation
            self.train_addrs = addrs[0:int(0.8 * len(addrs))]
            self.train_labels = labels[0:int(0.8 * len(labels))]
            self.val_addrs = addrs[int(0.8 * len(addrs)):int(len(addrs))]
            self.val_labels = labels[int(0.8 * len(addrs)):int(len(addrs))]



    def writeRecord(self,toBeRecorded,toBeLabeled,filename ):

        images_and_labels = []



        # train_filename = 'keremtrainwithsize.tfrecords'  # address to save the TFRecords file
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range(len(toBeRecorded)):
            # print how many images are saved every 1000 images
            if not i % 1000:
                print('Train data: {}/{}'.format(i, len(toBeRecorded)))
                sys.stdout.flush()
            # Load the image
            img = toBeRecorded[i]
            label = toBeLabeled[i]
            # Create a feature


            feature = {'label': _int64_feature(label),
                       'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()


    def read_decode_Record(self, batchsize=10,
                           data_path=None):  # bu şimdilik sadece resim ve label için, diğer featurelar için de eklenecek

        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}
        # Create a list of filenames and pass it to a que2
        # ue
        if data_path == None:
            data_path = self.filepath
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=None)

        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image'], tf.float32)

        # Cast label data into int32
        label = tf.cast(features['label'], tf.int32)
        # Reshape image data into the original shape
        image = tf.reshape(image, [227, 227, 3])

        # Any preprocessing here ...

        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batchsize,
                                                capacity=1000,
                                                num_threads=1,
                                                min_after_dequeue=10,
                                                allow_smaller_final_batch=False)
        return images, labels
