import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 0.1:
            print("\nLoss reached 0.1!")
            self.model.stop_training = True


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 2])
    plt.legend()
    plt.show()


def plot_comparision(test_labels, test_predictions):
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-10, 10], [-10, 10])
    plt.show()


def train(data, labels, epochs=1000):
    data, labels = shuffle(data, labels)

    num_train = int(0.7 * len(data))
    num_val = int(0.15 * len(data))

    train_data = data.iloc[:num_train]
    train_labels = labels.iloc[:num_train]

    val_data = data.iloc[num_train:num_train + num_val]
    val_labels = labels.iloc[num_train:num_train + num_val]

    test_data = data.iloc[num_train + num_val:]
    test_labels = labels.iloc[num_train + num_val:]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu',
                              input_shape=[len(train_data.keys())],
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])

    model.summary()

    history = model.fit(
        train_data, train_labels,
        batch_size=32,
        epochs=epochs,
        validation_data=(val_data, val_labels),
        callbacks=[CustomCallback()],
        verbose=2
    )

    plot_history(history)

    loss, mse = model.evaluate(test_data, test_labels)
    print('test loss: {}, test mse: {}'.format(loss, mse))

    test_predictions = model.predict(test_data).flatten()
    plot_comparision(test_labels, test_predictions)
