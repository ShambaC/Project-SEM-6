import tensorflow as tf

def residual_block(
        x,
        filter_num,
        strides = 2,
        kernel_size = 3,
        skip_convo = True,
        padding = 'same',
        kernel_initializer = 'he_uniform',
        dropout = 0.2
    ) :
        # Create skip connection tensor
        x_skip = x

        # Perform 1st convolution
        x = tf.keras.layers.Conv2D(filter_num, kernel_size, padding = padding, strides = strides, kernel_initializer = kernel_initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)

        # Perform 2nd convolution
        x = tf.keras.layers.Conv2D(filter_num, kernel_size, padding = padding, kernel_initializer = kernel_initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Perform 3rd convolution if skip_conv is True
        # Matches the number of filters and the shape of the skip connection tensor
        if skip_convo :
              x_skip = tf.keras.layers.Conv2D(filter_num, 1, padding = padding, strides = strides, kernel_initializer = kernel_initializer)(x_skip)

        # Add x and skip connection layers and apply activation function
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.LeakyReLU(0.1)(x)

        # Apply dropout
        if dropout :
              x = tf.keras.layers.Dropout(dropout)(x)

        return x


def train_model(input_dim, output_dim, dropout = 0.2) :

    inputs = tf.keras.layers.Input(shape = input_dim, name = "input")

    input = tf.keras.layers.Lambda(lambda x : x / 255)(inputs)

    x1 = residual_block(input, 16, skip_convo = True, strides = 1, dropout = dropout)

    x2 = residual_block(x1, 16, skip_convo = True, strides = 2, dropout = dropout)
    x3 = residual_block(x2, 16, skip_convo = False, strides = 1, dropout = dropout)

    x4 = residual_block(x3, 32, skip_convo = True, strides = 2, dropout = dropout)
    x5 = residual_block(x4, 32, skip_convo = False, strides = 1, dropout = dropout)

    x6 = residual_block(x5, 64, skip_convo = True, strides = 2, dropout = dropout)
    x7 = residual_block(x6, 64, skip_convo = True, strides = 1, dropout = dropout)

    x8 = residual_block(x7, 64, skip_convo = False, strides = 1, dropout = dropout)
    x9 = residual_block(x8, 64, skip_convo = False, strides = 1, dropout = dropout)

    squeeze = tf.keras.layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True))(squeeze)
    blstm = tf.keras.layers.Dropout(dropout)(blstm)

    output = tf.keras.layers.Dense(output_dim + 1, activation = 'softmax', name = "output")(blstm)

    model = tf.keras.Model(inputs = inputs, outputs = output)
    return model
