import keras.callbacks
import numpy as np

def default_lr_scheduler(
    base_lr = 0.01,
    steps = np.array([60000, 100000])
    ):

    def default_lr_scheduler_(iteration, lr):
        # stay on last lr
        if iteration >= steps[-1]:
            iteration = steps[-1] - 1

        if (iteration >= steps[0]) and (iteration < steps[1]):
            lr = base_lr / 10.0
        elif iteration >= steps[1]:
            lr = base_lr / 100.0
        return lr

    return default_lr_scheduler_


class LearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler (mostly copied from keras.callbacks.LearningRateScheduler).
    # Arguments
        schedule: a function that takes an iteration as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, base_lr=0.01, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule  = schedule
        self.iteration = 0
        self.verbose   = verbose

    def on_batch_begin(self, batch, logs=None):
        self.iteration += 1

        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        lr = self.schedule(self.iteration, lr=lr)

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be float (got {}).'.format(lr))

        keras.backend.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print()
            print('\nIteration {:05d}: LearningRateScheduler reducing learning rate to {}.'.format(self.iteration, lr))

class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.

    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```

    Args
        callback : callback to wrap.
        model    : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)
