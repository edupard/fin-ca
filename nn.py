from rbm import RBM
from au import AutoEncoder
from ffnn import FFNN
import tensorflow as tf
import numpy as np

TRAIN_RBM = False
RBM_EPOCH_TO_TRAIN = 50
RBM_BATCH_SIZE = 10

TRAIN_AU = False
LOAD_RBM_WEIGHTS = True
AU_EPOCH_TO_TRAIN = 30
AU_BATCH_SIZE = 10

TRAIN_FFNN = False
LOAD_AU_WEIGHTS = False
FFNN_EPOCH_TO_TRAIN = 1001
ALIGN_BATCH_TO_DATA = True
FFNN_BATCH_SIZE = 10
RANDOM_INIT = True
ALL_WEIGHTS_TRAINABLE = True

SAVE_EACH_N_EPOCHS = 500


def rbm_instance():
    rbmobject1 = RBM(17, 40, ['rbmw1', 'rbvb1', 'rbmhb1'], 0.001)
    rbmobject2 = RBM(40, 4, ['rbmw2', 'rbvb2', 'rbmhb2'], 0.001)
    return rbmobject1, rbmobject2


def ae_instance():
    autoencoder = AutoEncoder(17, [40, 4], [['rbmw1', 'rbmhb1'],
                                            ['rbmw2', 'rbmhb2']],
                              tied_weights=False)
    return autoencoder


def ffnn_instance():
    ffnn = FFNN(RANDOM_INIT, ALL_WEIGHTS_TRAINABLE, 17, [40, 4], [['rbmw1', 'rbmhb1'],
                                                                  ['rbmw2', 'rbmhb2']], transfer_function=tf.nn.sigmoid)
    return ffnn


def train_rbm(train_records, dr, wr):
    if not TRAIN_RBM:
        return
    # create rbm layers
    rbmobject1, rbmobject2 = rbm_instance()

    data_indices = np.arange(train_records)

    print("Training RBM layer 1")
    batches_per_epoch = train_records // RBM_BATCH_SIZE

    for i in range(RBM_EPOCH_TO_TRAIN):
        np.random.shuffle(data_indices)
        epoch_cost = 0.
        curr_progress = 0

        for b in range(batches_per_epoch):
            # get data indices for slice
            d_i_s = data_indices[b * RBM_BATCH_SIZE: (b + 1) * RBM_BATCH_SIZE]

            _wr = wr[d_i_s, :]
            _dr = dr[d_i_s, :]
            input = np.concatenate([_wr, _dr], axis=1)

            cost = rbmobject1.partial_fit(input)
            epoch_cost += cost
            progress = b // (batches_per_epoch // 10)
            if progress != curr_progress:
                print('.', sep=' ', end='', flush=True)
                curr_progress = progress
        print(" Epoch cost: {:.3f}".format(epoch_cost / batches_per_epoch))

    rbmobject1.save_weights('./rbm/rbmw1.chp')

    print("Training RBM layer 2")
    for i in range(RBM_EPOCH_TO_TRAIN):
        np.random.shuffle(data_indices)
        epoch_cost = 0.
        curr_progress = 0

        for b in range(batches_per_epoch):
            # get data indices for slice
            d_i_s = data_indices[b * RBM_BATCH_SIZE: (b + 1) * RBM_BATCH_SIZE]

            _wr = wr[d_i_s, :]
            _dr = dr[d_i_s, :]
            input = np.concatenate([_wr, _dr], axis=1)

            input = rbmobject1.transform(input)
            cost = rbmobject2.partial_fit(input)
            epoch_cost += cost
            progress = b // (batches_per_epoch // 10)
            if progress != curr_progress:
                print('.', sep=' ', end='', flush=True)
                curr_progress = progress
        print(" Epoch cost: {:.3f}".format(epoch_cost / batches_per_epoch))

    rbmobject2.save_weights('./rbm/rbmw2.chp')


def train_ae(train_records, dr, wr):
    if not TRAIN_AU:
        return
    autoencoder = ae_instance()

    print("Training Autoencoder")

    data_indices = np.arange(train_records)

    if LOAD_RBM_WEIGHTS:
        autoencoder.load_rbm_weights('./rbm/rbmw1.chp', ['rbmw1', 'rbmhb1'], 0)
        autoencoder.load_rbm_weights('./rbm/rbmw2.chp', ['rbmw2', 'rbmhb2'], 1)

    batches_per_epoch = train_records // AU_BATCH_SIZE
    for i in range(AU_EPOCH_TO_TRAIN):
        np.random.shuffle(data_indices)
        epoch_cost = 0.
        curr_progress = 0

        for b in range(batches_per_epoch):
            # get data indices for slice
            d_i_s = data_indices[b * AU_BATCH_SIZE: (b + 1) * AU_BATCH_SIZE]

            _wr = wr[d_i_s, :]
            _dr = dr[d_i_s, :]
            input = np.concatenate([_wr, _dr], axis=1)

            cost = autoencoder.partial_fit(input)
            # print("Batch cost: {:.3f}".format(cost))
            epoch_cost += cost
            progress = b // (batches_per_epoch // 10)
            if progress != curr_progress:
                print('.', sep=' ', end='', flush=True)
                curr_progress = progress
        print(" Epoch cost: {:.3f}".format(epoch_cost / batches_per_epoch))

    autoencoder.save_weights('./rbm/au.chp')


def train_ffnn(train_records, dr, wr, c_l, c_s, w_data_index, w_num_stocks):
    if not TRAIN_FFNN:
        return
    ffnn = ffnn_instance()

    print("Training FFNN")

    data_indices = np.arange(train_records)

    if LOAD_AU_WEIGHTS:
        ffnn.load_au_weights('./rbm/au.chp', ['rbmw1', 'rbmhb1'], 0)
        ffnn.load_au_weights('./rbm/au.chp', ['rbmw2', 'rbmhb2'], 1)

    if ALIGN_BATCH_TO_DATA:
        batches_per_epoch = w_data_index.shape[0]
        for i in range(FFNN_EPOCH_TO_TRAIN):
            epoch_cost = 0.
            curr_progress = 0

            for b in range(batches_per_epoch):
                s_i = w_data_index[b]
                e_i = s_i + w_num_stocks[b]
                _wr = wr[s_i:e_i, :]
                _dr = dr[s_i:e_i, :]
                input = np.concatenate([_wr, _dr], axis=1)

                _cl = c_l[s_i:e_i].reshape((-1, 1))
                _cs = c_s[s_i:e_i].reshape((-1, 1))
                observation = np.concatenate([_cl, _cs], axis=1).astype(np.float32)

                cost = ffnn.partial_fit(input, observation)
                # print("Batch cost: {:.3f}".format(cost))
                epoch_cost += cost
                progress = b // (batches_per_epoch // 10)
                if progress != curr_progress:
                    print('.', sep=' ', end='', flush=True)
                    curr_progress = progress
            print(" Epoch {} cost: {:.6f}".format(i, epoch_cost / batches_per_epoch))
            if i % SAVE_EACH_N_EPOCHS == 0:
                print("Model saved")
                ffnn.save_weights('./rbm/ffnn.chp')
    else:
        batches_per_epoch = train_records // FFNN_BATCH_SIZE
        for i in range(FFNN_EPOCH_TO_TRAIN):
            np.random.shuffle(data_indices)
            epoch_cost = 0.
            curr_progress = 0

            for b in range(batches_per_epoch):
                # get data indices for slice
                d_i_s = data_indices[b * FFNN_BATCH_SIZE: (b + 1) * FFNN_BATCH_SIZE]

                _wr = wr[d_i_s, :]
                _dr = dr[d_i_s, :]
                input = np.concatenate([_wr, _dr], axis=1)

                _cl = c_l[d_i_s].reshape((-1, 1))
                _cs = c_s[d_i_s].reshape((-1, 1))
                observation = np.concatenate([_cl, _cs], axis=1).astype(np.float32)

                cost = ffnn.partial_fit(input, observation)
                # print("Batch cost: {:.3f}".format(cost))
                epoch_cost += cost
                progress = b // (batches_per_epoch // 10)
                if progress != curr_progress:
                    print('.', sep=' ', end='', flush=True)
                    curr_progress = progress
            print(" Epoch {} cost: {:.6f}".format(i, epoch_cost / batches_per_epoch))
            if i % SAVE_EACH_N_EPOCHS == 0:
                print("Model saved")
                ffnn.save_weights('./rbm/ffnn.chp')

    ffnn.save_weights('./rbm/ffnn.chp')
    print("Model saved")


def evaluate_ffnn(data_set_records, dr, wr, prob_l):
    ffnn = ffnn_instance()

    ffnn.load_weights('./rbm/ffnn.chp')

    print("Evaluating")
    b = 0
    curr_progress = 0
    batches_per_epoch = data_set_records // FFNN_BATCH_SIZE
    while True:
        start_idx = b * FFNN_BATCH_SIZE
        end_idx = (b + 1) * FFNN_BATCH_SIZE
        d_i_s = np.arange(start_idx, min(end_idx, data_set_records))
        _wr = wr[d_i_s, :]
        _dr = dr[d_i_s, :]
        input = np.concatenate([_wr, _dr], axis=1)
        p_dist = ffnn.predict(input)
        for idx in d_i_s:
            prob_l[idx] = p_dist[idx - start_idx, 0]
        if end_idx >= data_set_records:
            break
        progress = b // (batches_per_epoch // 10)
        if progress != curr_progress:
            print('.', sep=' ', end='', flush=True)
            curr_progress = progress
        b += 1
