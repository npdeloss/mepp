import os
import shutil

import pandas as pd

from os.path import normpath
from slugify import slugify
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .io import (
    save_dataset
)

from .single import (
    wrap_single
)

orientation_to_filepath = {'+':'fwd','-':'rev','+/-':'fwd-rev'}
motif_id_and_orientation_to_filepath = lambda motif_id, orientation: (
    normpath(
        f'{slugify(motif_id)}'
        f'/orientation_'
        f'{orientation_to_filepath[orientation]}'
    )
)

def run_batch(
    dataset,
    motif_matrix_dict,
    out_filepath,
    center = None,
    motif_orientations = ['+', '-', '+/-'],
    motif_margin = 1,
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
    n_jobs = 1,
    free_dataset = True,
    keep_dataset = False,
    progress_wrapper = tqdm,
    no_gpu = False,
    stop_max_attempt_number = 3,
    wait_random_min = 1.0,
    wait_random_max = 2.0
):

    dataset_filepath = normpath(f'{out_filepath}/dataset')
    os.makedirs(
        dataset_filepath,
        exist_ok = True
    )

    save_dataset(
        dataset,
        dataset_filepath,
        delete_existing = True
    )
    if free_dataset:
        del dataset
    output_filepath_tups = [
        (
            motif_id,
            motif_orientation,
            normpath(
                f'{out_filepath}/' +
                motif_id_and_orientation_to_filepath(
                    motif_id,
                    motif_orientation
                )
            )
        )
        for motif_id
        in motif_matrix_dict
        for motif_orientation
        in motif_orientations
    ]

    wrap_single_params = [
        dict(
            dataset_filepath = dataset_filepath,
            motif_matrix = motif_matrix_dict[motif_id],
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
            save_figures = True,
            figure_formats = figure_formats,
            figure_dpi = figure_dpi,
            no_gpu = no_gpu,
            stop_max_attempt_number = stop_max_attempt_number,
            wait_random_min = wait_random_min,
            wait_random_max = wait_random_max
        )
        for motif_id, motif_orientation, output_filepath
        in output_filepath_tups
    ]

    if n_jobs > 1:
        n_jobs_ = n_jobs
        if no_gpu:
            # no_gpu mode still uses 2 threads per plot
            print(
                'Halving per-motif jobs for CPU-only operation.'
                'Two threads per MEPP plot will still be used.'
            )
            n_jobs_= n_jobs//2
            print(f'Number of per_motif jobs: {n_jobs_}')
        param_and_log_filepaths = Parallel(n_jobs = n_jobs)(
            delayed(wrap_single)(
                **wrap_single_param
            )
            for wrap_single_param
            in progress_wrapper(wrap_single_params)
        )
    else:
        param_and_log_filepaths = [
            wrap_single(
                **wrap_single_param
            )
            for wrap_single_param
            in progress_wrapper(wrap_single_params)
        ]


    output_filepath_tups = [
        [
            params[k]
            for k
            in [
                'motif_id',
                'motif_orientation',
                'output_filepath'
            ]
        ]
        for params
        in wrap_single_params
    ]

    filepath_tups = [
        tuple(a+list(b))
        for a, b
        in zip(
            output_filepath_tups,
            param_and_log_filepaths
        )
    ]

    filepaths_df = pd.DataFrame(
        filepath_tups,
        columns = [
            'motif_id',
            'orientation',
            'outdir',
            'params',
            'log',
            'retval',
            'attempts'
        ]
    )
    
    if keep_dataset == False:
        shutil.rmtree(normpath(dataset_filepath))
    
    return filepaths_df
