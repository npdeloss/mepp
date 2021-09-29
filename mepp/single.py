import os
import io
import sys
from os.path import normpath

import subprocess

from random import uniform
from time import sleep

import numpy as np

import yaml

import click

import matplotlib.pyplot as plt

from .io import (
    save_dataset,
    load_dataset
)

from .utils import (
    force_cpu_only,
    manage_gpu_memory
)

from .core import(
    dataset_and_motif_matrix_to_profile_data
)

from .plot import (
    plot_profile_data
)

@click.command()
@click.option(
    '-y',
    '--yaml',
    'yaml_filepath',
    required = True,
    type = str,
    help = 'Filepath to a YAML file describing the parameters for this MEPP.'
)
def run_single_from_yaml(yaml_filepath):
    """
    Handles Motif Enrichment Positional Profiling for a single motif.
    Currently used to bypass limitations with multiprocessing and Tensorflow.
    """
    if yaml_filepath is not None:
        with open(yaml_filepath, 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)
            params.pop('yaml_filepath', None)
            return run_single(**params)

def run_single(
    dataset_filepath = None,
    motif_matrix_filepath = None,
    motif_id = None,
    output_filepath = None,
    center = None,
    motif_orientation = '+',
    motif_margin = 0,
    motif_pseudocount = 0.0001,
    motif_pvalue = 0.0001,
    bg = None,
    confidence_interval_pct = 95,
    motif_score_sigma = 1.0,
    motif_score_cmap = 'gray',
    rank_smoothing_factor = 5,
    figure_width = 10,
    figure_height = 10,
    save_datasets = False,
    save_profile_data = True,
    save_motif_score_matrix = False,
    save_figures = True,
    figure_formats = ['png', 'svg'],
    figure_dpi = 300,
    no_gpu = False
):
    if no_gpu:
        force_cpu_only()
    else:
        manage_gpu_memory()

    params = dict(
        dataset_filepath = dataset_filepath,
        motif_matrix_filepath = motif_matrix_filepath,
        motif_id = motif_id,
        output_filepath = output_filepath,
        motif_orientation = motif_orientation,
        motif_margin = motif_margin,
        motif_pseudocount = motif_pseudocount,
        motif_pvalue = motif_pvalue,
        bg = bg,
        confidence_interval_pct = confidence_interval_pct,
        motif_score_sigma = motif_score_sigma,
        motif_score_cmap = motif_score_cmap,
        rank_smoothing_factor = rank_smoothing_factor,
        figure_width = figure_width,
        figure_height = figure_height,
        save_datasets = save_datasets,
        save_profile_data = save_profile_data,
        save_motif_score_matrix = save_motif_score_matrix,
        save_figures = save_figures,
        figure_formats = figure_formats,
        figure_dpi = figure_dpi
    )

    filepaths_ = dict(
        filepaths = (
            f'{output_filepath}/filepaths.yaml'
        ),
        params = (
            f'{output_filepath}/params.yaml'
        ),
        output = output_filepath,
        dataset = dataset_filepath,
        motif_matrix = motif_matrix_filepath,
        processed_dataset = (
            f'{output_filepath}/processed_dataset'
        ),
        smoothed_processed_dataset = (
            f'{output_filepath}/smoothed_processed_dataset'
        ),
        motif_score_matrix = (
            f'{output_filepath}/motif_score_matrix.npz'
        )
    )

    filepaths = {k:normpath(v) for k,v in filepaths_.items()}
    del filepaths_

    positions_df_filepaths_ = dict(
        tsv = f'{output_filepath}/positions_df.tsv',
        pickle = f'{output_filepath}/positions_df.pkl'
    )
    positions_df_filepaths = {
        k:normpath(v)
        for k,v
        in positions_df_filepaths_.items()
    }
    del positions_df_filepaths_
    filepaths['positions_df'] = positions_df_filepaths

    ranks_df_filepaths_ = dict(
        tsv = f'{output_filepath}/ranks_df.tsv',
        pickle = f'{output_filepath}/ranks_df.pkl'
    )
    ranks_df_filepaths = {
        k:normpath(v)
        for k,v
        in ranks_df_filepaths_.items()
    }
    del ranks_df_filepaths_
    filepaths['ranks_df'] = ranks_df_filepaths

    figure_filepaths_ = {
        fmt: f'{output_filepath}/mepp_plot.{fmt}'
        for fmt
        in figure_formats
    }
    figure_filepaths = {
        k:normpath(v)
        for k,v
        in figure_filepaths_.items()
    }
    del figure_filepaths_
    filepaths['mepp_plot'] = figure_filepaths

    os.makedirs(
        filepaths['output'],
        exist_ok = True
    )

    dataset = load_dataset(filepaths['dataset'])
    motif_matrix = np.load(filepaths['motif_matrix'])

    (
        motif_score_matrix,
        positions_df,
        ranks_df,
        processed_dataset,
        smoothed_processed_dataset
    ) = dataset_and_motif_matrix_to_profile_data(
        dataset,
        motif_matrix,
        orientation = motif_orientation,
        motif_margin = motif_margin,
        motif_pseudocount = motif_pseudocount,
        motif_pvalue = motif_pvalue,
        bg = bg,
        center = center,
        confidence_interval_pct = confidence_interval_pct
    )

    if save_datasets:
        save_dataset(
            processed_dataset,
            filepaths['processed_dataset'],
            delete_existing = True
        )

        save_dataset(
            smoothed_processed_dataset,
            filepaths['smoothed_processed_dataset'],
            delete_existing = True
        )
    else:
        filepaths.pop('processed_dataset', None)
        filepaths.pop('smoothed_processed_dataset', None)

    if save_profile_data:
        if save_motif_score_matrix:
            np.savez_compressed(
                filepaths['motif_score_matrix'],
                motif_score_matrix = motif_score_matrix
            )
        else:
            filepaths.pop('motif_score_matrix', None)

        positions_df.to_pickle(
            filepaths['positions_df']['pickle']
        )

        positions_df.to_csv(
            filepaths['positions_df']['tsv'],
            sep = '\t',
            index = False
        )

        ranks_df.to_pickle(
           filepaths['ranks_df']['pickle']
        )

        ranks_df.to_csv(
            filepaths['ranks_df']['tsv'],
            sep = '\t',
            index = False
        )
    else:
        filepaths.pop('motif_score_matrix', None)
        filepaths.pop('positions_df', None)
        filepaths.pop('ranks_df', None)

    if save_figures:
        plt.rcParams['figure.dpi'] = figure_dpi
        plt.rcParams['savefig.dpi'] = figure_dpi

    motif_matrix_stride = 1
    if motif_orientation == '-':
        motif_matrix_stride = -1

    fig, axes = plot_profile_data(
        motif_score_matrix,
        positions_df,
        ranks_df,
        motif_matrix = motif_matrix[
            ::motif_matrix_stride,
            ::motif_matrix_stride
        ],
        title = (
            f'Motif enrichment positional profile for \n '
             f'{motif_id} \n Orientation: {motif_orientation}'
        ),
        figsize = (figure_width, figure_height),
        rank_smoothing_factor = rank_smoothing_factor,
        motif_score_sigma = motif_score_sigma,
        motif_score_cmap = motif_score_cmap
    )

    for fmt in figure_formats:
        fig.savefig(
            filepaths['mepp_plot'][fmt],
            dpi = figure_dpi
        )
    plt.close(fig)

    with io.open(
        filepaths['params'],
        'w',
        encoding='utf8'
    ) as params_file:
        yaml.dump(
            params,
            params_file,
            default_flow_style=False,
            allow_unicode=True
        )

    with io.open(
        filepaths['filepaths'],
        'w',
        encoding='utf8'
    ) as filepaths_file:
        yaml.dump(
            filepaths,
            filepaths_file,
            default_flow_style=False,
            allow_unicode=True
        )

def generate_subprocess_args(yaml_filepath):
    # return ['python', '-m', 'mepp.single', '-y', yaml_filepath]
    return ['python', '-m', __name__, '-y', yaml_filepath]

def wrap_single(
    dataset_filepath,
    motif_matrix,
    motif_id,
    output_filepath,
    center = None,
    motif_orientation = '+',
    motif_margin = 0,
    motif_pseudocount = 0.0001,
    motif_pvalue = 0.0001,
    bg = None,
    confidence_interval_pct = 95,
    motif_score_sigma = 1.0,
    motif_score_cmap = 'gray',
    rank_smoothing_factor = 5,
    figure_width = 10,
    figure_height = 10,
    save_datasets = True,
    save_profile_data = True,
    save_motif_score_matrix = False,
    save_figures = True,
    figure_formats = ['png', 'svg'],
    figure_dpi = 300,
    no_gpu = False,
    subprocess_args_func = generate_subprocess_args,
    stop_max_attempt_number = 3,
    wait_random_min = 1.0,
    wait_random_max = 2.0
):
    """
    Wraps all motif-specific tasks in a subprocess call,
    enabling multiprocessing of motifs.
    """

    motif_matrix_filepath = normpath(f'{output_filepath}/motif_matrix.npy')

    params = dict(
        dataset_filepath = dataset_filepath,
        motif_matrix_filepath = motif_matrix_filepath,
        motif_id = motif_id,
        output_filepath = output_filepath,
        center = center,
        motif_orientation = motif_orientation,
        motif_margin = motif_margin,
        motif_pseudocount = motif_pseudocount,
        motif_pvalue = motif_pvalue,
        bg = bg,
        confidence_interval_pct = confidence_interval_pct,
        motif_score_sigma = motif_score_sigma,
        motif_score_cmap = motif_score_cmap,
        rank_smoothing_factor = rank_smoothing_factor,
        figure_width = figure_width,
        figure_height = figure_height,
        save_datasets = save_datasets,
        save_profile_data = save_profile_data,
        save_motif_score_matrix = save_motif_score_matrix,
        save_figures = save_figures,
        figure_formats = figure_formats,
        figure_dpi = figure_dpi,
        no_gpu = no_gpu
    )
    os.makedirs(
        normpath(output_filepath),
        exist_ok = True
    )

    np.save(
        motif_matrix_filepath,
        motif_matrix
    )

    yaml_filepath = normpath(f'{output_filepath}/wrapped_params.yaml')
    log_filepath = normpath(f'{output_filepath}/wrapped_params.log')
    with io.open(
        yaml_filepath,
        'w',
        encoding='utf8'
    ) as yaml_file:
        yaml.dump(
            params,
            yaml_file,
            default_flow_style=False,
            allow_unicode=True
        )
    if stop_max_attempt_number < 1:
        stop_max_attempt_number = 1
    attempt = 1
    while attempt <= stop_max_attempt_number:
        with open(os.devnull, 'w') as f, open(log_filepath, 'w') as lf:
            retval = subprocess.call(
                generate_subprocess_args(yaml_filepath),
                stderr=lf,
                stdout=f
            )
            if retval != 0:
                attempt += 1
                sleep(uniform(wait_random_min, wait_random_max))
            else:
                break
    attempts = attempt
    return yaml_filepath, log_filepath, retval, attempts


if __name__ == '__main__':
    sys.exit(run_single_from_yaml())
