"""Main module."""

import os
import multiprocessing

from os.path import normpath

from .io import (
    scored_fasta_filepath_to_dicts,
    motif_matrix_filepath_to_dicts,
    save_dataset,
    load_dataset
)
from .utils import (
    force_cpu_only,
    manage_gpu_memory,
    order_scored_fasta_df,
    filter_scored_fasta_df,
    scored_fasta_dicts_to_df,
    append_gc_ratios_to_dataset,
    scored_fasta_df_to_dataset,
    filepaths_df_to_profile_dicts
)

from .single import (
    wrap_single
)

from .batch import (
    run_batch,
    orientation_to_filepath
)

from .clustering import (
    cluster_profiles
)


from .permutations import (
    add_permuted_scores_to_dataset
)

from .clustering import (
    filepaths_df_to_clustering_results
)

def run_mepp(
    # Filepaths
    scored_fasta_filepath,
    motifs_filepath,
    out_filepath,
    # Dataset parameters
    center = None,
    degenerate_pct_thresh = 100,
    num_permutations = 1000,
    batch_size = 1000,
    n_cpu_jobs = multiprocessing.cpu_count(),
    # Motif parameters
    motif_orientations = ['+','+/-'],
    motif_margin = 2,
    motif_pseudocount = 0.0001,
    motif_pvalue = 0.0001,
    bg = None,
    # Figure parameters
    confidence_interval_pct = 95,
    motif_score_sigma = 0.5,
    motif_score_cmap = 'gray',
    rank_smoothing_factor = 5,
    figure_width = 10,
    figure_height = 10,
    figure_formats = ['png', 'svg'],
    figure_dpi = 300,
    n_gpu_jobs = 3,
    # Save parameters
    save_datasets = False,
    save_profile_data = True,
    # Save/Don't save big motif score matrix AKA raw heatmap
    save_motif_score_matrix = False,
    keep_dataset = False,
    # Retry parameters
    stop_max_attempt_number = 3,
    wait_random_min = 1.0,
    wait_random_max = 2.0,
    # Clustering parameters
    cluster_method = 'average',
    cluster_metric = 'correlation',
    mepp_plot_format = 'png',
    table_image_dpi = 100,
    table_image_format = 'png',
    mt_method = 'fdr_tsbky',
    mt_alpha = 0.01,
    thorough_mt = True,
    no_gpu = False
):
    # dendrogram_level = num_clusters
    gc_ratio_type = 'global'

    # Designate dataset filepaths
    filtered_data_df_pkl_filepath = normpath(
        f'{out_filepath}/filtered_data_df.pkl'
    )
    filtered_data_df_tsv_filepath = normpath(
        f'{out_filepath}/filtered_data_df.tsv'
    )
    dataset_filepath = normpath(f'{out_filepath}/dataset')
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

    scored_fasta_df.to_csv(
        filtered_data_df_tsv_filepath,
        sep = '\t',
        index = False
    )

    scored_fasta_df.to_pickle(filtered_data_df_pkl_filepath)

    # Create Tensorflow dataset from scored sequence data
    dataset = add_permuted_scores_to_dataset(
        append_gc_ratios_to_dataset(
            scored_fasta_df_to_dataset(
                scored_fasta_df,
                batch_size = batch_size,
                n_jobs = n_cpu_jobs
            ),
            gc_ratio_type = gc_ratio_type
        ),
        batch_size = batch_size,
        num_permutations = num_permutations
    )

    # Load motif matrices
    (
        motif_matrix_dict,
        motif_consensus_dict
    ) = motif_matrix_filepath_to_dicts(
        motifs_filepath
    )

    # Run batch processing
    filepaths_df = run_batch(
        dataset = dataset,
        motif_matrix_dict = motif_matrix_dict,
        out_filepath = out_filepath,
        center = center,
        motif_orientations = motif_orientations,
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
        keep_dataset = keep_dataset,
        save_profile_data = save_profile_data,
        save_motif_score_matrix = save_motif_score_matrix,
        save_figures = True,
        figure_formats = figure_formats,
        figure_dpi = figure_dpi,
        n_jobs = n_gpu_jobs,
        no_gpu = no_gpu,
        stop_max_attempt_number = stop_max_attempt_number,
        wait_random_min = stop_max_attempt_number,
        wait_random_max = stop_max_attempt_number,
    )

    filepaths_filepath = normpath(f'{out_filepath}/filepaths.tsv')

    filepaths_df.to_csv(filepaths_filepath, sep = '\t', index = False)
    #
    # # Make dendrogram for each motif orientation
    # heatmap_html_filepaths = {}
    # for motif_orientation in motif_orientations:
    #     # Set filepaths of outputs
    #     orientation_str = orientation_to_filepath[motif_orientation]
    #     heatmap_html_filepath = normpath(
    #         f'{out_filepath}/heatmap_orientation_{orientation_str}.html'
    #     )
    #     heatmap_df_pkl_filepath = normpath(
    #         f'{out_filepath}/heatmap_orientation_{orientation_str}.pkl'
    #     )
    #     heatmap_df_tsv_filepath = normpath(
    #         f'{out_filepath}/heatmap_orientation_{orientation_str}.tsv'
    #     )
    #
    #     # Map motif IDs to profiles
    #     profile_dicts = filepaths_df_to_profile_dicts(
    #         filepaths_df,
    #         motif_orientation,
    #         mepp_plot_format = figure_formats[0]
    #     )
    #     motif_id_to_profile = profile_dicts['motif_id_to_profile']
    #     motif_id_to_mepp_plot = profile_dicts['motif_id_to_mepp_plot']
    #     motif_id_to_profile_df = profile_dicts['motif_id_to_profile_df']
    #
    #     # Cluster profiles
    #     dendrogram_depth = dendrogram_level
    #     (
    #         clustered_profiles_df,
    #         linkage_matrix,
    #         linkage_tree_root
    #     ) = cluster_profiles(
    #         motif_id_to_profile,
    #         dendrogram_depth,
    #         method = 'average',
    #         metric = 'correlation'
    #     )
    #
    #     # HTML format heatmap dataframe
    #     styled_df_style, styled_df = style_clustered_profiles_df(
    #         clustered_profiles_df,
    #         motif_id_to_mepp_plot = re_prefix_filepath_dict(
    #             motif_id_to_mepp_plot,
    #             f'{out_filepath}/',
    #             ''
    #         )
    #     )
    #
    #     # Write heatmap and styled dataframe to file
    #     clustered_profiles_df.to_csv(
    #         heatmap_df_tsv_filepath,
    #         sep = '\t',
    #         index = False
    #     )
    #     clustered_profiles_df.to_pickle(heatmap_df_pkl_filepath)
    #     with open(heatmap_html_filepath, 'w') as html_file:
    #         html_file.write(
    #             f'<html><head></head><body>'
    #             f'{styled_df_style.render()}</body></html>'
    #         )
    #     heatmap_html_filepaths[motif_orientation] = heatmap_html_filepath
    #
    # return filepaths_df, heatmap_html_filepaths

    # Generate clustering outputs
    results_html_filepaths = {}
    clustermap_html_filepaths = {}
    for motif_orientation in motif_orientations:
        orientation_str = orientation_to_filepath[motif_orientation]
        clustering_results = filepaths_df_to_clustering_results(
            out_filepath,
            center = center,
            motif_orientation = motif_orientation,
            cluster_method = cluster_method,
            cluster_metric = cluster_metric,
            mepp_plot_format = mepp_plot_format,
            mt_method = mt_method,
            mt_alpha = mt_alpha,
            thorough_mt = thorough_mt,
            table_image_dpi = table_image_dpi,
            table_image_format = table_image_format,
            n_jobs = n_cpu_jobs
        )
        table_html_filepath = normpath(f'{out_filepath}/results_table_orientation_{orientation_str}.html')
        clustermap_html_filepath = normpath(f'{out_filepath}/clustermap_orientation_{orientation_str}.html')
        table_pkl_filepath = normpath(f'{out_filepath}/results_table_orientation_{orientation_str}.pkl')
        table_tsv_filepath = normpath(f'{out_filepath}/results_table_orientation_{orientation_str}.tsv')
        clustering_results['results_df'].to_csv(table_tsv_filepath, sep = '\t', index = False)
        clustering_results['results_df'].to_pickle(table_pkl_filepath)
        with open(table_html_filepath, 'w') as html_file:
            html_file.write(clustering_results['results_html'])
        results_html_filepaths[motif_orientation] = table_html_filepath

        with open(clustermap_html_filepath, 'w') as html_file:
            html_file.write(clustering_results['clustermap_html'])
        clustermap_html_filepaths[motif_orientation] = clustermap_html_filepath

    return filepaths_df, results_html_filepaths, clustermap_html_filepaths
