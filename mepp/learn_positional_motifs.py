"""Helper script to learn motifs given positional profiles using neural networks"""

import os
import sys
import click
import multiprocessing

from os.path import normpath

from contextlib import redirect_stdout

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

from .learn_motifs import (
    revcomp_augment_dataset, 
    revcomp_dataset, 
    reformat_dataset
)

from .learn_motifs import (
    amplify_motifs, 
    extract_motifs, 
    filter_motifs_by_information_content, 
    motifs_to_dict, 
    motif_matrix_dict_to_file
)

def positionalize_dataset_scores(dataset, positional_profile):
    return dataset.map(
        lambda sequences, scores: (
            sequences, 
            (
                tf.expand_dims(
                    tf.cast(
                        positional_profile, 
                        scores.dtype
                    ), 
                    axis=0
                ) * 
                scores
            )
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

def generate_positional_conv_model(
    input_layer, 
    num_motifs = 320, 
    motif_length = 8, 
    seed = 10, 
    motif_margin=2, 
    unstranded = True
):
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
    
    revcomp_conv_model = keras.Model(
        inputs = input_layer,
        outputs = (
            conv_model(input_layer[:,::-1,::-1])
            [:,::-1,:]
        )
    )

    dual_conv_model = keras.Model(
        inputs = input_layer,
        outputs = tf.math.reduce_max(
            [
                conv_model(input_layer), 
                revcomp_conv_model(input_layer)
            ]
            , axis = 0
        )
    )
    
    if unstranded:
        chosen_conv_model = dual_conv_model
    else:
        chosen_conv_model = conv_model
    
    pool_size = 1+motif_margin*2
    post_conv_model = keras.Sequential([
        input_layer,
        chosen_conv_model,
        keras.layers.AveragePooling1D(
            pool_size, 
            strides = 1
        )
    ])
  
    total_pad = input_layer.shape[1]-post_conv_model.output_shape[1]
    left_pad = total_pad//2
    right_pad = total_pad-left_pad
    
    model = keras.Sequential([
        input_layer,
        post_conv_model,
        keras.layers.ZeroPadding1D(padding=(left_pad, right_pad)),
#         keras.layers.Conv1D(
#             num_motifs, 
#             1,
#             groups=num_motifs,
#             activation='linear', 
#             kernel_initializer = keras.initializers.GlorotUniform(seed=seed+2),
#             kernel_regularizer = 'l1_l2',
#             use_bias = False
#         )
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
    return model, conv_model, post_conv_model


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
    '--profile',
    'profile_filepath',
    type = str,
    required = True,
    help = (
        'Path to a tab-separated text file '
        'with two columns: position, and profile'
        'describing positions along a sequence '
        'and the target profile for enriching a motif '
        'at those positions. '
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
# Positional parameters
@click.option(
    '--center',
    'center',
    type = int,
    default = None,
    help = (
        '0-based offset from the start of the sequence to center profiles on. '
        'Default: Set the center to half the sequence length, rounded down'
    )
)
@click.option(
    '--margin',
    'motif_margin',
    type = int,
    default = 2,
    help = (
        'Number of bases along either side of motif to "blur" motif matches '
        'for smoothing. '
        'Default: 2'
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
    '--orientation',
    'orientation',
    type = click.Choice([
        '+',
        '-',
        '+/-'
    ]),
    default = '+',
    help = (
        'Orientation of motifs. '
        '+ and - are equivalent. '
        '+/- enables strand-invariant motif learning, '
        'which is useful if the input data are unstranded.'
        'Default: +'
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
    profile_filepath,
    out_filepath,
    # Positional parameters
    center = None,
    motif_margin = 2,
    # Dataset parameters
    degenerate_pct_thresh = 100.0,
    batch_size = 1000,
    validation_fraction = 0.10,
    # Motif parameters
    num_motifs = 320,
    motif_length = 8,
    orientation = '+',
    motif_prefix = 'positional_denovo_motif_',
    # Model parameters
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
    with redirect_stdout(sys.stderr):
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
        # Normalize scores
        scored_fasta_df['original_score'] = scored_fasta_df['score']
        
        scored_fasta_df['z_score'] = (
            (scored_fasta_df['original_score'] - scored_fasta_df['original_score'].mean()) /
            (scored_fasta_df['original_score'].std())
        )
        
        scored_fasta_df['score'] = scored_fasta_df['z_score']
        
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
        num_validation_batches = np.max([int(np.round(validation_fraction * num_batches)), 1])
        num_training_batches = num_batches - num_validation_batches
        

        # print(f'{num_validation_batches} validation batches')
        # print(f'{num_training_batches} training batches')
        
        if np.min([num_validation_batches, num_training_batches]) < 1:
            print(
                (
                    'Not enough batches to have validation and training datasets. '
                    'Re-run analysis with a smaller batch size.'
                ),
                file = sys.stderr
            )
        
        # Load profile
        profile_df = pd.read_csv(
            profile_filepath,
            sep = '\t'
        )
        
        if 'positional_r' in list(profile_df.columns):
            profile_df['profile'] = profile_df['positional_r']
        
        
        # Align profile to current dataset positions
        max_sequence_length = scored_fasta_df['sequence'].map(len).max()
        positions = np.array(list(range(max_sequence_length)))
        
        if center is None:
            center = max_sequence_length//2
        positions = (positions - center)
        
        positions_df = pd.DataFrame({'position':positions})
        
        profile_df = (
            positions_df
            .merge(profile_df[['position', 'profile']], how = 'left')
            .bfill()
            .ffill()
        )
        
        # Extract positional profile to scale with scores
        positional_r_vec = tf.constant(profile_df['profile'])
        positional_profile = positional_r_vec
        
        # Create dataset
        original_dataset = scored_fasta_df_to_dataset(
            padded_scored_fasta_df,
            batch_size = batch_size,
            n_jobs = 1
        )

#         revcomp_augmented_dataset = revcomp_augment_dataset(
#             original_dataset, 
#             batch_size = batch_size, 
#             random_seed = seed
#         )
#         dataset_rev = revcomp_dataset(
#             revcomp_augmented_dataset, 
#             batch_size = batch_size
#         )
#         dataset = (
#             revcomp_augmented_dataset
#             .concatenate(dataset_rev)
#             .prefetch(tf.data.AUTOTUNE)
#             .cache()
#         )
        
        # No reverse complementation
        dataset = (
            original_dataset
            .prefetch(tf.data.AUTOTUNE)
            .cache()
        )

        
        # Save dataset
        save_dataset(dataset, dataset_filepath)

        # Split dataset into validation and training
        validation_dataset = (
            positionalize_dataset_scores(
                reformat_dataset(
                    load_dataset(dataset_filepath)
                ), 
                positional_profile
            )
            .take(num_validation_batches) 
        )
        training_dataset = (
            positionalize_dataset_scores(
                reformat_dataset(
                    load_dataset(dataset_filepath)
                ), 
                positional_profile
            )
            # .skip(num_validation_batches)
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
        
        unstranded = False
        if orientation is '+/-':
            print('Assuming data is unstranded. Will learn motifs with orientation invariance.')
            unstranded = True
        
        model_type = 'positional_conv'
        generate_model = generate_positional_conv_model
        model_name = f'{model_type}_seed_{seed}_{num_motifs}_motifs_{motif_length}bp_margin_{motif_margin}'
        model, conv_model, post_conv_model = generate_model(
            input_layer, 
            num_motifs, 
            motif_length, 
            seed,
            motif_margin,
            unstranded
        )

        # Get model output dimensions
        num_model_output_dims = len(list(model.layers[-1].output_shape))
        num_score_dims = len(list(scores.shape))

        # Match model output dimensions
        for i in list(range(num_model_output_dims-num_score_dims)):
            validation_dataset = validation_dataset.map(
                lambda sequence, score: (sequence, tf.expand_dims(score, -1)), 
                num_parallel_calls=tf.data.AUTOTUNE
            )
            training_dataset = training_dataset.map(
                lambda sequence, score: (sequence, tf.expand_dims(score, -1)), 
                num_parallel_calls=tf.data.AUTOTUNE
            )


        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE).cache()
        training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE).cache()

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
                monitor = 'loss',
                patience = early_stopping_patience, 
                min_delta = early_stopping_min_delta
            ))


        print('Training model', file = sys.stderr )
        model.fit(
            training_dataset,
            epochs=epochs,
            # validation_data=validation_dataset,
            callbacks=callbacks,
            verbose = 'auto',
        )

        # Extract motifs
        print('Extracting motifs', file = sys.stderr )
        motifs_, information_matrices = amplify_motifs(
            extract_motifs(
                conv_model, 
                dataset, 
                n_jobs = n_jobs, 
                unstranded = unstranded
            )
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
    
    with redirect_stdout(sys.stdout):
        if quiet == False:
            with open(denovo_motifs_filepath) as f:
                print(f.read(), file = sys.stdout)
    
    with redirect_stdout(sys.stderr):
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
