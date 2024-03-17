import numpy as np
import tensorflow as tf


def get_weight_matrix(layers, input, seed):
    weight_matrix = []
    for i, n in enumerate(layers):
        rows, columns = (n, input.shape[1]+1) if i == 0 else (n, layers[i-1]+1)
        np.random.seed(seed)
        x = np.random.randn(columns, rows).astype("float32")
        weight_matrix.append(tf.Variable(x))
    return weight_matrix


def get_weight_matrix0(weights):
    weight_matrix = []
    for x in weights:
        weight_matrix.append(tf.Variable(x, dtype="float32"))
    return weight_matrix


def forward_pass(weight_matrix, input, activations):
    input = tf.convert_to_tensor(input)
    input = tf.transpose(input)
    for (weights, transferfunction) in zip(weight_matrix, activations):
        tensor_shape = tf.shape(input)
        ones_row = tf.ones(shape=(1, tensor_shape[1]), dtype="float32")
        input = tf.cast(input, tf.float32)
        input = tf.concat([ones_row, input], axis=0)
        net = tf.matmul(tf.transpose(input), weights)
        if transferfunction == "sigmoid":
            input = tf.nn.sigmoid(net)
        elif transferfunction == "linear":
            input = net
        elif transferfunction == "relu":
            input = tf.nn.relu(net)
        input = tf.transpose(input)
    return tf.transpose(input)


def split_data(X_train, Y_train, split_range):
    start = int(split_range[0] * X_train.shape[0])
    end = int(split_range[1] * X_train.shape[0])
    return np.concatenate((X_train[:start], X_train[end:])), np.concatenate(
        (Y_train[:start], Y_train[end:])), X_train[start:end], Y_train[start:end]


def generate_batches(X, y, batch_size):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
    # if there's any data left, yield it
    if X.shape[0] % batch_size != 0:
        yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]


def loss_crsntrpy(target_y, predicted_y):
    sparse_ce = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_y, logits=predicted_y)
    return tf.reduce_mean(sparse_ce)


def loss_mse(target_y, predicted_y):
    mse_loss = tf.reduce_mean(tf.square(target_y - predicted_y))
    return mse_loss


def loss_svm(target_y, predicted_y):
    margin = 1.0
    hinge_loss = tf.reduce_mean(tf.maximum(
        0.0, margin - target_y * predicted_y))
    return hinge_loss


def fit(x_train, y_train, learning_rate, epochs, batch_size, weight_matrix, activations, X_val, y_val, loss):
    '''function uses stochastic gradient descent to train a neural network. The training data, learning rate, 
    number of epochs, batch size, weight matrix, activation functions, validation data, 
    and kind of loss function to be employed are all inputs to the function. The weight matrix 
    is then subjected to gradient descent for the specified number of epochs, after which a 
    list of the estimated loss values for each epoch is returned. Mean squared error, cross-entropy, 
    and support vector machine loss are the three different types of loss functions that the function supports.
    '''
    err = []
    loss_functions = {
        'mse': loss_mse,
        'cross_entropy': loss_crsntrpy,
        'svm': loss_svm
    }
    loss_fn = loss_functions[loss]

    for epoch_index in range(epochs):
        for X_batch, y_batch in generate_batches(x_train, y_train, batch_size):
            with tf.GradientTape(persistent=True) as t:
                t.watch(weight_matrix)
                prediction = forward_pass(weight_matrix, X_batch, activations)
                loss = loss_fn(y_batch, prediction)
                gradiente = t.gradient(loss, weight_matrix)

            for weight, gradient in zip(weight_matrix, gradiente):
                weight.assign_sub(learning_rate * gradient)

        prediction = forward_pass(weight_matrix, X_val, activations)
        test_loss = loss_fn(y_val, prediction)
        err.append(test_loss)

    return err


def multi_layer_nn_tensorflow(X_train, Y_train, layers, activations, alpha, batch_size, epochs=1, loss="svm",
                              validation_split=[0.8, 1.0], weights=None, seed=2):
    if (weights):
        weights = get_weight_matrix0(weights)

    else:
        weights = get_weight_matrix(layers, X_train, seed)
    X_train_mod, y_train_mod, X_val, y_val = split_data(
        X_train, Y_train, validation_split)
    err = fit(X_train_mod, y_train_mod, alpha, epochs, batch_size,
              weights, activations, X_val, y_val, loss)
    Out = forward_pass(weights, X_val, activations)

    return weights, err, Out
