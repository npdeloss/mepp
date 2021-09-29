import math
import numpy as np

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import InputLayer, Conv1D, Activation, AveragePooling1D, Flatten, ZeroPadding1D
from tensorflow.math import reduce_max, reduce_sum

import MOODS.tools

from .utils import get_normalize_onehot_model

def motif_matrix_to_conv1d_layer(
    motif_matrix,
    sequence_length,
    revcomp = False,
    dual = False,
    pseudocount = 0.0001,
    pvalue = 0.0001,
    bg = None,
    padding = 'valid'
):
    if bg is None:
        bg = MOODS.tools.flat_bg(motif_matrix.shape[0])

    motif_length = motif_matrix.shape[1]

    conv_layer_input = Input(
        shape = [
            sequence_length,
            motif_matrix.shape[0]
        ]
    )

    orientation_int = -1 if revcomp else 1

    motif_matrix_weight = np.array(
        MOODS.tools.log_odds(
            motif_matrix.tolist(),
            bg,
            pseudocount
        )
    )[::orientation_int,::orientation_int]

    motif_matrix_bias = MOODS.tools.threshold_from_p(
        motif_matrix_weight,
        bg,
        pvalue
    )

    filters = 1 if dual != True else 2

    motif_conv = Conv1D(
      filters = filters,
      kernel_size = motif_length,
      activation = 'linear',
      padding = padding,
      trainable = False
    )

    motif_conv(conv_layer_input)

    weights, biases = motif_conv.get_weights()

    if dual != True:
        motif_matrix_weights = np.stack([
            motif_matrix_weight
        ]).T
    else:
        motif_matrix_weights = np.stack([
            motif_matrix_weight,
            motif_matrix_weight[::-1,::-1]
        ]).T

    motif_matrix_biases = np.zeros(biases.shape) - motif_matrix_bias

    motif_conv.set_weights([motif_matrix_weights, motif_matrix_biases])

    return motif_conv

def motif_matrix_to_conv_model(
    motif_matrix,
    sequence_length,
    activation = 'relu',
    revcomp = False,
    dual = False,
    pseudocount = 0.0001,
    pvalue = 0.0001,
    bg = None,
    normalize_onehot = True,
    padding = 'valid'
):

    input_shape = (
        sequence_length,
        motif_matrix.shape[0]
    )

    
    if padding == 'valid':
        unpadded_output_length = math.floor((sequence_length - motif_matrix.shape[1]) / 1) + 1
        pad_left = (sequence_length - unpadded_output_length)//2
        pad_right = sequence_length - unpadded_output_length - pad_left
        padding_vals = (pad_left, pad_right)
    else:
        padding_vals = (0, 0)
    
    padding_layer = ZeroPadding1D(padding = padding_vals)
    
    conv_layer_input = Input(
        shape = input_shape
    )

    if normalize_onehot:
        normalize_onehot_model = get_normalize_onehot_model(
                conv_layer_input
        )


    conv_model_input_layer = InputLayer(
        input_shape = input_shape
    )

    if dual != True:
        motif_conv_model_1 = motif_matrix_to_conv1d_layer(
            motif_matrix,
            sequence_length,
            revcomp = revcomp,
            dual = False,
            pseudocount = pseudocount,
            pvalue = pvalue,
            bg = bg,
            padding = padding
        )
    else:
        motif_conv_model_0 = motif_matrix_to_conv1d_layer(
            motif_matrix,
            sequence_length,
            revcomp = revcomp,
            dual = True,
            pseudocount = pseudocount,
            pvalue = pvalue,
            bg = bg,
            padding = padding
        )
        motif_conv_model_1 = Model(
            inputs = conv_layer_input,
            outputs = reduce_max(
                motif_conv_model_0(conv_layer_input),
                axis = -1,
                keepdims = True
            )
        )

    activation_layer = Activation(activation)

    motif_conv_model_2_layers = [conv_model_input_layer]
    if normalize_onehot:
        motif_conv_model_2_layers.append(normalize_onehot_model)
    motif_conv_model_2_layers.append(motif_conv_model_1)
    motif_conv_model_2_layers.append(padding_layer)
    motif_conv_model_2_layers.append(activation_layer)

    motif_conv_model_2 = Sequential(motif_conv_model_2_layers)

    return motif_conv_model_2
