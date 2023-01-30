import math
import tensorflow as tf


def ae(inputSize, codeSize,
       corr=0.0, tight=False,
       enc=tf.nn.tanh, dec=tf.nn.tanh):

    x = tf.placeholder(tf.float32, [None, inputSize])

    if corr > 0.0:
        _x = tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                      minval=0,
                                                      maxval=1 - corr,
                                                      dtype=tf.float32), tf.float32))

    else:
        _x = x

    b_enc = tf.Variable(tf.zeros([codeSize]))

    W_enc = tf.Variable(tf.random_uniform(
                [inputSize, codeSize],
                -6.0 / math.sqrt(inputSize + codeSize),
                6.0 / math.sqrt(inputSize + codeSize))
            )

    encode = tf.matmul(_x, W_enc) + b_enc
    if enc is not None:
        encode = enc(encode)

    b_dec = tf.Variable(tf.zeros([inputSize]))
    if tight:

        W_dec = tf.transpose(W_enc)

    else:

        W_dec = tf.Variable(tf.random_uniform(
                    [codeSize, inputSize],
                    -6.0 / math.sqrt(codeSize + inputSize),
                    6.0 / math.sqrt(codeSize + inputSize))
                )

    decode = tf.matmul(encode, W_dec) + b_dec
    if dec is not None:
        decode = enc(decode)

    model = {

        "input": x,

        "encode": encode,

        "decode": decode,

        "cost": tf.sqrt(tf.reduce_mean(tf.square(x - decode))),

        "params": {
            "W_enc": W_enc,
            "b_enc": b_enc,
            "b_dec": b_dec,
        }

    }

    if not tight:
        model["params"]["W_dec"] = W_dec

    return model


def nn(inputSize, N, layers, init=None):

    input = x = tf.placeholder(tf.float32, [None, inputSize])

    y = tf.placeholder("float", [None, N])

    actvs = []
    dropouts = []
    params = {}

    for i, layer in enumerate(layers):

        dropout = tf.placeholder(tf.float32)

        if init is None:
            W = tf.Variable(tf.zeros([inputSize, layer["size"]]))
            b = tf.Variable(tf.zeros([layer["size"]]))

        else:
            W = tf.Variable(init[i]["W"])
            b = tf.Variable(init[i]["b"])

        x = tf.matmul(x, W) + b
        if "actv" in layer and layer["actv"] is not None:
            x = layer["actv"](x)

        x = tf.nn.dropout(x, dropout)

        params.update({
            "W_" + str(i+1): W,
            "b_" + str(i+1): b,
        })
        actvs.append(x)
        dropouts.append(dropout)

        inputSize = layer["size"]

    W = tf.Variable(tf.random_uniform(
            [inputSize, N],
            -3.0 / math.sqrt(inputSize + N),
            3.0 / math.sqrt(inputSize + N)))
    b = tf.Variable(tf.zeros([N]))

    y_hat = tf.matmul(x, W) + b

    params.update({"W_out": W, "b_out": b})
    actvs.append(y_hat)

    return {

        "input": input,

        "expected": y,

        "output": tf.nn.softmax(y_hat),

        "cost": tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y)),

        "dropouts": dropouts,

        "actvs": actvs,

        "params": params,
    }
