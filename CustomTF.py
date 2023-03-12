import tensorflow as tf

class CTCloss(tf.keras.losses.Loss) :
    def __init__(self, name = 'CTCloss'):
        super(CTCloss, self).__init__()
        self.name = name
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def __call__(self, y_true, y_pred, sample_weight = None) -> tf.Tensor :
        batch_len = tf.cast(tf.shape(y_true)[0], dtype = "int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype = "int64")
        label_length = tf.cast(tf.shape(y_pred)[1], dtype = "int64")

        input_length = input_length * tf.ones(shape = (batch_len, 1), dtype= "int64")
        label_length = label_length * tf.ones(shape = (batch_len, 1), dtype= "int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        return loss
    
class CWERMetric(tf.keras.metrics.Metric) :
    def __init__(self, padding_token, name='CWER', **kwargs) :
        super(CWERMetric, self).__init__(name = name, **kwargs)

        self.cer_accumulator = tf.Variable(0.0, name= "cer_accumulator", dtype= tf.float32)
        self.wer_accumulator = tf.Variable(0.0, name= "wer_accumulator", dtype= tf.float32)
        self.batch_counter = tf.Variable(0, name= "batch_counter", dtype= tf.int32)

        self.padding_token = padding_token

    def update_state(self, y_true, y_pred, sample_weight = None) :
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape= input_shape[0], dtype='int32') * tf.cast(input_shape[1], 'int32')

        decode_predicted, log = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

        predicted_labels_sparse = tf.keras.backend.ctc_label_dense_to_sparse(decode_predicted[0], input_length)

        true_labels_sparse = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(y_true, input_length), "int64")

        predicted_labels_sparse = tf.sparse.retain(predicted_labels_sparse, tf.not_equal(predicted_labels_sparse.values, -1))

        distance = tf.edit_distance(predicted_labels_sparse, true_labels_sparse, normalize=True)

        self.cer_accumulator.assign_add(tf.reduce_sum(distance))

        self.batch_counter.assign_add(len(y_true))

        self.wer_accumulator.assign_add(tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32)))

    def result(self) :
        return {
            "CER" : tf.math.divide_no_nan(self.cer_accumulator, tf.cast(self.batch_counter, tf.float32)),
            "WER" : tf.math.divide_no_nan(self.wer_accumulator, tf.cast(self.batch_counter, tf.float32))
        }
    
import numpy as np
import copy
import pandas as pd
import cv2
    
class DataProvider(tf.keras.utils.Sequence) :
    def __init__(
            self,
            dataset,
            vocab,
            max_text_length,
            batch_size = 16,
            initial_epoch = 1
    ) :
        # super.__init__()
        self._dataset = dataset
        self._batch_size = batch_size
        self._epoch = initial_epoch
        self._step = 0
        self._vocab = vocab
        self._max_text_length = max_text_length

    def __len__(self) :
        return int(np.ceil(len(self._dataset)/ self._batch_size))

    @property
    def epoch(self) :
        return self._epoch
    
    @property
    def step(self) :
        return self._step
    
    def on_epoch_end(self) :
        self._epoch += 1

    def split(self, split = 0.9, shuffle = True) :
        if shuffle :
            np.random.shuffle(self._dataset)

        train_data = copy.deepcopy(self)
        val_data = copy.deepcopy(self)
        train_data._dataset = self._dataset[:int(len(self._dataset) * split)]
        val_data._dataset = self._dataset[int(len(self._dataset) * split):]

        return train_data, val_data
    
    def to_csv(self, path, index = False) :
        df = pd.DataFrame(self._dataset)
        df.to_csv(path, index= index)

    def get_batch_annotations(self, index) :
        self._step = index
        start_index = index * self._batch_size

        batch_indexes = [i for i in range(start_index, start_index + self._batch_size) if i < len(self._dataset)]

        batch_annotations = [self._dataset[index] for index in batch_indexes]

        return batch_annotations
    
    # Transformers
    def ImageResizer(self, data, label) :
        return cv2.resize(data, (128, 32), interpolation= cv2.INTER_AREA), label
    
    def LabelIndexer(self, data, label) :
        return data, np.array([self._vocab.index(l) for l in label if l in self._vocab])
    
    def LabelPadding(self, data, label) :
        return data, np.pad(label, (0, self._max_text_length - len(label)), 'constant', constant_values= len(self._vocab))
    
    def __getitem__(self, index) :
        dataset_batch = self.get_batch_annotations(index)

        batch_data, batch_annotations = [], []
        for index, (data, annotation) in enumerate(dataset_batch) :
            data = cv2.imread(data, cv2.IMREAD_COLOR)

            if data is None :
                self._dataset.remove(dataset_batch[index])
                continue

            batch_data.append(data)
            batch_annotations.append(annotation)

        batch_data, batch_annotations = zip(*[self.ImageResizer(data, annotation) for data, annotation in zip(batch_data, batch_annotations)])
        batch_data, batch_annotations = zip(*[self.LabelIndexer(data, annotation) for data, annotation in zip(batch_data, batch_annotations)])
        batch_data, batch_annotations = zip(*[self.LabelPadding(data, annotation) for data, annotation in zip(batch_data, batch_annotations)])

        return np.array(batch_data), np.array(batch_annotations)