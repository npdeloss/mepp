"""Helper script to learn motifs using neural networks"""

import os
import sys
import click
import multiprocessing

from os.path import normpath

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from logomaker import transform_matrix

from .io import (
    scored_fasta_filepath_to_dicts,
    save_dataset,
    load_dataset
)
from .utils import (
    force_cpu_only,
    manage_gpu_memory,
    order_scored_fasta_df,
    filter_scored_fasta_df,
    scored_fasta_dicts_to_df,
    scored_fasta_df_to_dataset,
)

from .plot import motif_matrix_to_df
from .html import get_logo_df
from .io import motif_matrix_file_to_dicts
from .html import motif_matrix_to_logo_data_uri
from itertools import combinations

def revcomp_augment_dataset(dataset, batch_size = 1000, random_seed = 10):
    rng = np.random.default_rng(random_seed)
    augmented_dataset = (
        dataset.unbatch()
        .map(lambda sequence, score: (rng.integers(1,endpoint=True)*2-1, sequence, score), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda strand, sequence, score: (sequence[::strand,::strand],score), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .cache()
    )
    return augmented_dataset

def revcomp_dataset(dataset, batch_size = 1000):
    augmented_dataset = (
        dataset.unbatch()
        .map(lambda sequence, score: (-1, sequence, score), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda strand, sequence, score: (sequence[::strand,::strand],score), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .cache()
    )
    return augmented_dataset

def reformat_dataset(dataset):
    return dataset.map(
        lambda sequences, scores: (
            tf.cast(sequences, float), 
            tf.expand_dims(scores,-1)
        ), 
        num_parallel_calls=tf.data.AUTOTUNE
    )

def generate_deepbind_model(input_layer, num_motifs = 320, motif_length = 8, seed = 10):
    conv_model = keras.Sequential([
        input_layer,
        keras.layers.Conv1D(
            num_motifs, 
            motif_length,activation='relu',
            kernel_initializer = keras.initializers.GlorotUniform(seed=seed)
        ),
    ])
    post_conv_model = keras.Sequential([
        input_layer,
        conv_model,
        keras.layers.MaxPool1D(4,4),
        keras.layers.Dropout(0.2),
        keras.layers.Conv1D(
            480,8,
            activation='relu',
            kernel_initializer = keras.initializers.GlorotUniform(seed=seed+1)
        ),
        keras.layers.MaxPool1D(4,4),
        keras.layers.Dropout(0.2),
        keras.layers.Conv1D(
            960,8,
            activation='relu',
            kernel_initializer = keras.initializers.GlorotUniform(seed=seed+2)
        ),
        keras.layers.MaxPool1D(4,4),
        keras.layers.Dropout(0.5),
    ])

    flatten = keras.layers.Flatten()
    dense = keras.layers.Dense(
        1,
        activation='tanh',
        kernel_initializer = keras.initializers.GlorotUniform(seed=seed+3)
    )

    model = keras.Sequential([
        input_layer,
        post_conv_model,
        flatten,
        dense
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model, conv_model, post_conv_model

def generate_simpleconv_model(input_layer, num_motifs = 320, motif_length = 8, seed = 10):
    conv_model = keras.Sequential([
        input_layer,
        keras.layers.Conv1D(
            32, 
            2,activation='relu', 
            kernel_initializer = keras.initializers.GlorotUniform(seed=seed)
        ),
        keras.layers.Conv1D(
            num_motifs, 
            motif_length-1,activation='relu', 
            kernel_initializer = keras.initializers.GlorotUniform(seed=seed+1)
        )
    ])
    post_conv_model = keras.Sequential([
        input_layer,
        conv_model
    ])
    
    flatten = keras.layers.Flatten()
    dense = keras.layers.Dense(
        1,
        activation='tanh',
        kernel_initializer = keras.initializers.GlorotUniform(seed=seed+2)
    )

    model = keras.Sequential([
        input_layer,
        post_conv_model,
        flatten,
        dense
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model, conv_model, post_conv_model

model_generation_by_model_type = {
    'deepbindlike': generate_deepbind_model,
    'simpleconv': generate_simpleconv_model
}

@tf.autograph.experimental.do_not_convert
def get_activations(conv_model, sequences, scores):
    activations = conv_model(sequences, training = False)
    max_idxs = keras.backend.argmax(activations,axis = 1)
    max_vals = keras.backend.max(activations,axis = 1)
    return (
        # sequences
        sequences, 
        # activations
        activations,
        # maximum activation indices
        max_idxs,
        # maximum activation values
        max_vals
    )

def get_weighted_subsequence_sums(
        sequences, 
        activations,
        max_idxs,
        max_vals
):
    num_sequences = sequences.shape[0]
    num_motifs = activations.shape[-1]
    motif_length = sequences.shape[1]-activations.shape[1]+1
    motif_width = sequences.shape[-1]
    
    zero_motif = np.zeros((motif_length,motif_width))
    
    # print(type(sequences))
    subsequences = np.stack([
        np.stack([
            sequence[motif_max_idx:motif_max_idx+motif_length,:]
            for motif_max_idx
            in sequence_max_idxs
        ])
        for sequence, sequence_max_idxs
        in zip(sequences, max_idxs)
    ], axis = 0)
    
    weighted_subsequence_sums = np.sum(
        subsequences * np.expand_dims(np.expand_dims(max_vals,-1),-1),
        axis = 0
    )
    
    return weighted_subsequence_sums

def extract_motifs(conv_model, dataset, n_jobs = multiprocessing.cpu_count()):
    dataset_activations = (
        dataset
        .map(
            lambda sequences, scores: get_activations(conv_model, sequences, scores),
        )
    )

    weighted_subsequence_sums = np.sum(Parallel(
        n_jobs=n_jobs
    )(
        delayed(get_weighted_subsequence_sums)(
            sequences.numpy(), 
            activations.numpy(),
            max_idxs.numpy(),
            max_vals.numpy()
        ) 
        for sequences, 
            activations,
            max_idxs,
            max_vals
        in tqdm(dataset_activations)
    ), axis = 0)

    eps = np.finfo(weighted_subsequence_sums.dtype).eps
    motifs =  (weighted_subsequence_sums+eps) / (np.sum(weighted_subsequence_sums, axis = -1, keepdims = True)+eps)
    return motifs

def amplify_motifs(motifs):
    information_matrices = np.stack([
        transform_matrix(
            motif_matrix_to_df(m.T), 
            from_type = 'probability',
            to_type = 'information'
        ) 
        for m in motifs
    ])
    eps = np.finfo(information_matrices.dtype).eps
    motif_extra_power = (
        (2.0+eps)/
        (
            information_matrices
            .sum(axis=-1,keepdims = True)
            .max(axis=-2,keepdims = True) + eps
        )
    )

    amplified_weighted_subsequence_sums = np.power(motifs,motif_extra_power)
    amplified_motifs =  (
        (amplified_weighted_subsequence_sums) / 
        (np.sum(
            amplified_weighted_subsequence_sums, axis = -1, keepdims = True
        ))
    )

    return amplified_motifs, information_matrices

def filter_motifs_by_information_content(motifs, information_matrices, min_information_content = 0.0):
    max_info_contents = information_matrices.sum(axis = -1).sum(axis = -1)
    return np.stack([
        motif
        for motif, info_content
        in zip(motifs, max_info_contents)
        if info_content > min_information_content
    ])

def motifs_to_dict(motifs, motif_prefix = 'denovo_motif_'):
    motif_matrix_dict = {
        f'{motif_prefix}{i+1}':motif.T 
        for i, motif 
        in enumerate(motifs)
    }
    return motif_matrix_dict

def motif_matrix_dict_to_file(motif_matrix_dict, f):
    f.write('\n'.join([
        f'>{motif_id}\n'+'\n'.join([
            '\t'.join([
                f'{e:.3f}' for e in row
            ]) for row in motif_matrix
        ]) 
        for motif_id, motif_matrix 
        in motif_matrix_dict.items()
    ])+'\n')


@click.command()
# Filepaths
@click.option(
    '--fa',
    'scored_fasta_filepath',
    type = str,
    required = True,
    help = (
        'Path to a scored fasta file, '
        'where sequence headers are of the form: '
        '">sequence_name sequence_score".'
    )
)
@click.option(
    '--out',
    'out_filepath',
    type = str,
    required = True,
    help = (
        'Create this directory and write output to it.'
    )
)
# Dataset parameters
@click.option(
    '--dgt',
    'degenerate_pct_thresh',
    type = float,
    default = 100.0,
    help = (
        'Percentage of sequence that can be degenerate '
        '(Not A, C, G, or T) before being rejected from the analysis. '
        'Useful for filtering out repeats. '
        'Default: 100'
    )
)

@click.option(
    '--batch',
    'batch_size',
    type = int,
    default = 1000,
    help = (
        'Size of batches for Tensorflow datasets. '
        'Default: 1000'
    )
)
@click.option(
    '--val',
    'validation_fraction',
    type = float,
    default = 0.10,
    help = (
        'Fraction of data used for validation. '
        'Default: 0.10'
    )
)
# Motif parameters
@click.option(
    '--motifs',
    'num_motifs',
    type = int,
    default = 320,
    help = (
        'Number of motifs to learn. '
        'Default: 320'
    )
)
@click.option(
    '--length',
    'motif_length',
    type = int,
    default = 8,
    help = (
        'Length of motifs to learn. '
        'Default: 8'
    )
)
@click.option(
    '--motif-prefix',
    'motif_prefix',
    type = str,
    default = 'denovo_motif_',
    help = (
        'Prefix motif names with this string.'
        'Default: denovo_motif_'
    )
)
# Model parameters
@click.option(
    '--model',
    'model_type',
    type = click.Choice(
        [
            'deepbindlike',
            'simpleconv'
        ],
        case_sensitive =  False
    ),
    default = 'deepbindlike',
    help = (
        'Type of network to use for learning motifs. '
        'Default: deepbindlike'
    )
)
@click.option(
    '--seed',
    'seed',
    type = int,
    default = 1000,
    help = (
        'Random seed for shuffling and initialization. '
        'Default: 1000'
    )
)
# Training parameters
@click.option(
    '--epochs',
    'epochs',
    type = int,
    default = 1000,
    help = (
        'Maximum number of epochs for training. '
        'Default: 1000'
    )
)
@click.option(
    '--no-early-stopping',
    'early_stopping',
    default = True,
    flag_value = False,
    help = (
        'Disable early stopping of training, to train for the maximum number of epochs. '
        'Default: Enable early stopping.'
    )
)
@click.option(
    '--patience',
    'early_stopping_patience',
    type = int,
    default = 10,
    help = (
        'Number of epochs to wait for early stopping. '
        'Default: 1000'
    )
)
@click.option(
    '--mindelta',
    'early_stopping_min_delta',
    type = float,
    default = 0.0,
    help = (
        'Minimum delta for early stopping. '
        'Default: 0'
    )
)
# Execution parameters
@click.option(
    '--jobs',
    'n_jobs',
    type = int,
    default = multiprocessing.cpu_count(),
    help = (
        'Number of jobs '
        'for CPU multiprocessing. '
        'Default: Use all cores'
    )
)
@click.option(
    '--nogpu',
    'no_gpu',
    is_flag=True,
    help = (
        'Disable use of GPU. '
    )
)
@click.option(
    '--quiet',
    'quiet',
    default = False,
    is_flag = True,
    flag_value = True,
    help = (
        'Do not write combined motifs to stdout. '
        'Default: Write combined motifs to stdout.'
    )
)
def main(
    # Filepaths
    scored_fasta_filepath,
    out_filepath,
    # Dataset parameters
    degenerate_pct_thresh = 100.0,
    batch_size = 1000,
    validation_fraction = 0.10,
    # Motif parameters
    num_motifs = 320,
    motif_length = 8,
    motif_prefix = 'denovo_motif_',
    # Model parameters
    model_type = 'deepbindlike',
    seed = 10,
    # Training parameters
    epochs = 1000,
    early_stopping = True,
    early_stopping_patience = 5,
    early_stopping_min_delta = 0,
    # Execution parameters
    n_jobs = multiprocessing.cpu_count(),
    no_gpu = False,
    quiet = False
):
    # Setup output

    filtered_data_df_tsv_filepath = normpath(f'{out_filepath}/filtered_data_df.tsv')
    filtered_data_df_pkl_filepath = normpath(f'{out_filepath}/filtered_data_df.pkl')
    dataset_filepath = normpath(f'{out_filepath}/dataset')

    os.makedirs(
        normpath(out_filepath), 
        exist_ok = True
    )

    os.makedirs(
        dataset_filepath,
        exist_ok = True
    )
    
    # Manage GPU memory usage

    if no_gpu:
        force_cpu_only()
    else:
        manage_gpu_memory()
        
    # Load scored sequence data
    (
        sequence_dict,
        score_dict,
        description_dict
    ) = scored_fasta_filepath_to_dicts(scored_fasta_filepath)

    # Filter and sort sequence data
    scored_fasta_df = order_scored_fasta_df(
        filter_scored_fasta_df(
            scored_fasta_dicts_to_df(
                sequence_dict,
                score_dict,
                description_dict
            ),
            degenerate_pct_thresh = degenerate_pct_thresh
        )
    )
    
    # Write filtered data to file
    scored_fasta_df.to_csv(
        filtered_data_df_tsv_filepath,
        sep = '\t',
        index = False
    )

    scored_fasta_df.to_pickle(filtered_data_df_pkl_filepath)
    
    # Pad to meet batch size
    scored_fasta_df_padding = (
        scored_fasta_df.sample(
            n = (batch_size - scored_fasta_df.shape[0]%batch_size), 
            random_state = seed
        )
        .copy()
    )
    padded_scored_fasta_df = (
        pd.concat([
            scored_fasta_df, 
            scored_fasta_df_padding
        ])
        .sample(frac = 1.0, random_state = seed)
        .copy()
        .reset_index()
    )
    
    # Get validation and training batch amounts
    num_batches = padded_scored_fasta_df.shape[0]//batch_size
    num_validation_batches = np.min([int(np.round(validation_fraction * num_batches)), 1])
    num_training_batches = num_batches - num_validation_batches
    
    # Create dataset
    original_dataset = scored_fasta_df_to_dataset(
        padded_scored_fasta_df,
        batch_size = batch_size,
        n_jobs = 1
    )
    
    revcomp_augmented_dataset = revcomp_augment_dataset(
        original_dataset, 
        batch_size = batch_size, 
        random_seed = seed
    )
    dataset_rev = revcomp_dataset(
        revcomp_augmented_dataset, 
        batch_size = batch_size
    )
    dataset = (
        revcomp_augmented_dataset
        .concatenate(dataset_rev)
        .prefetch(tf.data.AUTOTUNE)
        .cache()
    )
    
    # Save dataset
    save_dataset(dataset, dataset_filepath)
    
    # Split dataset into validation and training
    validation_dataset = (
        reformat_dataset(load_dataset(dataset_filepath))
        .take(num_validation_batches) 
    )
    training_dataset = (
        reformat_dataset(load_dataset(dataset_filepath))
        .skip(num_validation_batches)
    )
    
    # Determine input layer properties
    sequences, scores = (
        reformat_dataset(load_dataset(dataset_filepath))
        .take(1)
        .get_single_element()
    )
    input_layer = (
        keras.layers.Input(
            type_spec = tf.TensorSpec.from_tensor(sequences)
        )
    )
    
    # Generate model
    
    generate_model = model_generation_by_model_type[model_type]
    model_name = f'{model_type}_seed_{seed}_{num_motifs}_motifs_{motif_length}bp'
    model, conv_model, post_conv_model = generate_model(
        input_layer, 
        num_motifs, 
        motif_length, 
        seed
    )
    
    # Train model
    callbacks = [
        keras.callbacks.experimental.BackupAndRestore(
            normpath(f'{out_filepath}/{model_name}.model_backups/')
        ),
        keras.callbacks.ModelCheckpoint(
            normpath(f'{out_filepath}/{model_name}.model_checkpoints/')
        ),
        keras.callbacks.CSVLogger(
            normpath(f'{out_filepath}/{model_name}.training_log.csv')
        )
    ]
    
    if early_stopping:
        print('Training with early stopping', file = sys.stderr )
        callbacks.append(keras.callbacks.EarlyStopping(
            patience = early_stopping_patience, 
            min_delta = early_stopping_min_delta
        ))
    
    
    print('Training model', file = sys.stderr )
    model.fit(
        training_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose = 0,
    )
    
    # Extract motifs
    print('Extracting motifs', file = sys.stderr )
    motifs_, information_matrices = amplify_motifs(
        extract_motifs(conv_model, dataset, n_jobs = n_jobs)
    )
    motifs = filter_motifs_by_information_content(
        motifs_, 
        information_matrices, 
        min_information_content = 0.0
    )
    
    # Format motifs
    motif_matrix_dict = motifs_to_dict(motifs, motif_prefix)
    
    # Write motifs to file
    print('Writing motifs', file = sys.stderr )
    denovo_motifs_filepath = normpath(
        f'{out_filepath}/{model_name}.denovo_motifs.txt'
    )
    
    with open(denovo_motifs_filepath, 'w') as f:
        motif_matrix_dict_to_file(motif_matrix_dict, f)
    
    if quiet == False:
        with open(denovo_motifs_filepath) as f:
            print(f.read())
    
    # Write motif logos to file
    print('Writing motif logos', file = sys.stderr )
    motif_matrix_df = get_logo_df(motif_matrix_dict, n_jobs = n_jobs)
    denovo_motifs_html_filepath = normpath(
        f'{out_filepath}/{model_name}.denovo_motifs.html'
    )
    
    denovo_motifs_logo_filepath = normpath(
        f'{out_filepath}/{model_name}.denovo_motifs.pkl'
    )
    
    header_include_html = '\n'.join([
        '<link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.0.1/css/bootstrap.min.css"/>'
        '<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.0.1/js/bootstrap.bundle.min.js"></script>'
    
    ])
    title = f'Logos for {denovo_motifs_filepath}'
    head_html = f'<meta charset="UTF-8"><title>{title}</title>{header_include_html}'
    html = (
        f'''<!DOCTYPE html><head>{head_html}</head>'''
        f'''<body>{motif_matrix_df.style.render()}</body></html>'''
        f'\n'
    )
    
    with open(denovo_motifs_html_filepath, 'w') as f:
        f.write(html)
    
    motif_matrix_df.to_pickle(denovo_motifs_logo_filepath)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover    