"""Console script for mepp."""
import sys
import click

import multiprocessing

import matplotlib.pyplot as plt

from .mepp import run_mepp

@click.command()
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
    '--motifs',
    'motifs_filepath',
    type = str,
    required = True,
    help = (
        'Path to a motif matrices file in JASPAR format.'
        'As a start, one can be obtained through the JASPAR website at: '
        'http://jaspar.genereg.net/downloads/'
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
@click.option(
    '--center',
    'center',
    type = int,
    default = None,
    help = (
        '0-based offset from the start of the sequence to center plots on. '
        'Default: Set the center to half the sequence length, rounded down'
    )
)
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
    '--perms',
    'num_permutations',
    type = int,
    default = 1000,
    help = (
        'Number of permutations for permutation testing '
        'and confidence intervals. '
        'Can lead to significant GPU memory usage. '
        'Default: 1000'
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
    '--jobs',
    'n_cpu_jobs',
    type = int,
    default = multiprocessing.cpu_count(),
    help = (
        'Number of jobs '
        'for CPU multiprocessing. '
        'Default: Use all cores'
    )
)
@click.option(
    '--keepdata', 
    'keep_dataset',
    is_flag=True,
    help = (
        'Set this flag to keep the Tensorflow dataset '
        'after MEPP has finished. '
        'Default: Delete the dataset after MEPP has finished. '
    )
)
@click.option(
    '--orientations',
    'motif_orientations',
    type = str,
    default = '+,+/-',
    help = (
        'Comma-separated list of motif orientations to analyze '
        'for CPU multiprocessing. '
        'Values in list are limited to '
        '"+" (Match motif forward orientation), '
        '"-" (Match motif reverse orientation), '
        '"+/-" (Match to forward or reverse). '
        'Default: +,+/-'
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
@click.option(
    '--pcount',
    'motif_pseudocount',
    type = float,
    default = 0.0001,
    help = (
        'Pseudocount for setting motif match threshold via MOODS. '
        'Default: 0.0001'
    )
)
@click.option(
    '--pval',
    'motif_pvalue',
    type = float,
    default = 0.0001,
    help = (
        'P-value for setting motif match threshold via MOODS. '
        'Default: 0.0001'
    )
)
@click.option(
    '--bg',
    'bg',
    nargs = 4,
    type = float,
    default = (0.25, 0.25, 0.25, 0.25),
    help = (
        'Background DNA composition, for setting motif match threshold '
        'via MOODS, '
        'represented as a series of 4 floats. '
        'Default: 0.25 0.25 0.25 0.25'
    )
)
@click.option(
    '--ci',
    'confidence_interval_pct',
    type = float,
    default = 95.0,
    help = (
        'Confidence interval for positional profile, expressed as a percentage. '
        'Default: 95.0'
    )
)
@click.option(
    '--sigma',
    'motif_score_sigma',
    type = float,
    default = 0.5,
    help = (
        'Adaptive scale for brightness of motif matches in motif heatmaps. '
        'Maximum brightness is achieved at sigma * std, '
        'where std is the standard deviation of nonzero motif match scores. '
        'Set lower for brighter pixels. '
        'Must be a positive value. '
        'Default: 0.5'
    )
)
@click.option(
    '--cmap',
    'motif_score_cmap',
    type = str,
    default = 'gray_r',
    help = (
        'Name of a matplotlib colormap. '
        'Used to color the central MEPP motif heatmap. '
        'Possible values can be viewed using '
        'matplotlib.pylot.colormaps() or at '
        'https://matplotlib.org/stable/tutorials/colors/colormaps.html . '
        'Default: gray_r. Set to gray to invert colors (black background).'
    )
)
@click.option(
    '--smoothing',
    'rank_smoothing_factor',
    type = int,
    default = 5,
    help = (
        'Factor by which to smooth motif density along ranks '
        'for visualization. '
        'This is multiplicative to smoothing that already occurs dependent on '
        'figure pixel resolution. '
        'Default: 5'
    )
)
@click.option(
    '--width',
    'figure_width',
    type = int,
    default = 10,
    help = (
        'Width of generated MEPP plot, in inches. '
        'Default: 10'
    )
)
@click.option(
    '--height',
    'figure_height',
    type = int,
    default = 10,
    help = (
        'Height of generated MEPP plot, in inches. '
        'Default: 10'
    )
)
@click.option(
    '--formats',
    'figure_formats',
    type = str,
    default = 'png,svg',
    help = (
        'Comma-separated list of image formats for MEPP plots. '
        'Possible formats are png and svg. '
        'Default: png,svg'
    )
)
@click.option(
    '--dpi',
    'figure_dpi',
    type = int,
    default = 300,
    help = (
        'DPI of generated MEPP plot. '
        'Default: 300'
    )
)
@click.option(
    '--gjobs',
    'n_gpu_jobs',
    type = int,
    default = 1,
    help = (
        'Number of jobs '
        'for GPU multiprocessing. '
        'NOTE: Set this carefully to avoid jobs crowding each other '
        'out of GPU memory, causing profile generation to fail. '
        'If setting --nogpu, this will be the number of jobs used to '
        'process motifs in parallel. '
        'Default: 1'
    )
)
@click.option(
    '--nogpu',
    'no_gpu',
    is_flag=True,
    help = (
        'Disable use of GPU. '
        'If setting --nogpu, --gjobs will be the number of jobs used to '
        'process motifs in parallel.'
    )
)
@click.option(
    '--attempts',
    'stop_max_attempt_number',
    type = int,
    default = 10,
    help = (
        'Number of attempts to retry making a plot. '
        'Default: 10'
    )
)
@click.option(
    '--minwait',
    'wait_random_min',
    type = float,
    default = 1.0,
    help = (
        'Minimum wait between attempts to make a plot, in seconds. '
        'Default: 1.0'
    )
)
@click.option(
    '--maxwait',
    'wait_random_max',
    type = float,
    default = 1.0,
    help = (
        'Maximum wait between attempts to make a plot, in seconds. '
        'Default: 1.0'
    )
)
@click.option(
    '--cmethod',
    'cluster_method',
    type = click.Choice(
        [
            'average',
            'single',
            'complete',
            'centroid',
            'median',
            'ward',
            'weighted'
        ],
        case_sensitive =  False
    ),
    default = 'average',
    help = (
        'Clustering method for clustering MEPP profiles. '
        'For details, see "method" parameter of '
        'scipy.cluster.hierarchy.linkage. '
        'Default: average'
    )
)
@click.option(
    '--cmetric',
    'cluster_metric',
    type = click.Choice(
        [
            'correlation',
            'euclidean',
            'minkowski',
            'cityblock',
            'seuclidean',
            'sqeuclidean',
            'cosine',
            'jensenshannon',
            'chebyshev',
            'canberra',
            'braycurtis',
            'mahalanobis'
        ],
        case_sensitive = False
    ),
    default = 'correlation',
    help = (
        'Clustering metric for clustering MEPP profiles. '
        'For details, see "metric" parameter of '
        'scipy.cluster.hierarchy.linkage. '
        'Default: correlation'
    )
)
@click.option(
    '--tdpi',
    'table_image_dpi',
    type = int,
    default = 100,
    help = (
        'DPI of inline plots for clustering table. '
        'Default: 100'
    )
)
@click.option(
    '--tformat',
    'table_image_format',
    type= click.Choice(
        [
            'png',
            'svg'
        ],
        case_sensitive = False
    ),
    default = 'png',
    help = (
        'Format of inline plots for clustering table. '
        'Use png for speed, svg for publication quality. '
        'Default: png'
    )
)
@click.option(
    '--mtmethod',
    'mt_method',
    type= click.Choice(
        [
            'fdr_tsbky',
            'bonferroni',
            'sidak',
            'holm-sidak',
            'holm',
            'simes-hochberg',
            'hommel',
            'fdr_bh',
            'fdr_by',
            'fdr_tsbh'
        ],
        case_sensitive = False
    ),
    default = 'fdr_by',
    help = (
        'Multiple testing method for adjusting p-values of '
        'positional correlations listed in the clustering table.'
        'For details, see "method" parameter of '
        'statsmodels.stats.multitest.multipletests. '
        'Default: fdr_by'
    )
)
@click.option(
    '--mtalpha',
    'mt_alpha',
    type = float,
    default = 0.01,
    help = (
        'Alpha (FWER, family-wise error rate) for adjusting p-values of '
        'positional correlations listed in the clustering table.'
        'For details, see "alpha" parameter of '
        'statsmodels.stats.multitest.multipletests. '
        'Default: 0.01'
    )
)
@click.option(
    '--thoroughmt',
    'thorough_mt',
    flag_value = True,
    default = True,
    help = (
        'Enables thorough multiple testing '
        'of positional correlation p-values: '
        'All p-values for all motifs '
        'at all positions will be adjusted simultaneously.'
        'Default: Thorough multiple testing is enabled'
    )
)
@click.option(
    '--non-thoroughmt',
    'thorough_mt',
    flag_value = False,
    default = True,
    help = (
        'Disables thorough multiple testing '
        'of positional correlation p-values: '
        'Only extreme p-values will be adjusted for.'
        'Default: Thorough multiple testing is enabled'
    )
)
def main(
    # Filepaths
    scored_fasta_filepath,
    motifs_filepath,
    out_filepath,
    # Dataset parameters
    center = None,
    degenerate_pct_thresh = 100.0,
    num_permutations = 1000,
    batch_size = 1000,
    n_cpu_jobs = multiprocessing.cpu_count(),
    keep_dataset = False,
    # Motif parameters
    motif_orientations = '+,+/-',
    motif_margin = 2,
    motif_pseudocount = 0.0001,
    motif_pvalue = 0.0001,
    bg = None,
    # Figure parameters
    confidence_interval_pct = 95.0,
    motif_score_sigma = 0.5,
    motif_score_cmap = 'gray',
    rank_smoothing_factor = 5,
    figure_width = 10,
    figure_height = 10,
    figure_formats = 'png,svg',
    figure_dpi = 300,
    n_gpu_jobs = 3,
    no_gpu = False,
    # Retry parameters
    stop_max_attempt_number = 10,
    wait_random_min = 1.0,
    wait_random_max = 2.0,
    # Clustering parameters
    cluster_method = 'average',
    cluster_metric = 'correlation',
    table_image_dpi = 100,
    table_image_format = 'png',
    mt_method = 'fdr_tsbky',
    mt_alpha = 0.01,
    thorough_mt = True
):
    """
    Profile positional enrichment of motifs in a list of scored sequences.
    Generates MEPP (Motif Enrichment Positional Profile) plots.
    """
    motif_orientations_ = [
        x.strip()
        for x
        in motif_orientations.split(',')
    ]
    bg_ = list(bg)
    figure_formats_ = [
        x.strip().lower()
        for x
        in figure_formats.split(',')
    ]
    mepp_plot_format = figure_formats_[0]

    motif_score_cmap_ = motif_score_cmap
    if motif_score_cmap not in set(plt.colormaps()):
        motif_score_cmap_ = 'gray'

    n_gpu_jobs_ = n_gpu_jobs
    # if no_gpu:
    #     n_gpu_jobs_ = 1

    (
        filepaths_df,
        results_html_filepaths,
        clustermap_html_filepaths,
    ) = run_mepp(
        # Filepaths
        scored_fasta_filepath,
        motifs_filepath,
        out_filepath,
        # Dataset parameters
        center = center,
        degenerate_pct_thresh = degenerate_pct_thresh,
        num_permutations = num_permutations,
        batch_size = batch_size,
        n_cpu_jobs = n_cpu_jobs,
        keep_dataset = keep_dataset,
        # Motif parameters
        motif_orientations = motif_orientations_,
        motif_margin = motif_margin,
        motif_pseudocount = motif_pseudocount,
        motif_pvalue = motif_pvalue,
        bg = bg_,
        # Figure parameters
        confidence_interval_pct = confidence_interval_pct,
        motif_score_sigma = motif_score_sigma,
        motif_score_cmap = motif_score_cmap_,
        rank_smoothing_factor = rank_smoothing_factor,
        figure_width = figure_width,
        figure_height = figure_height,
        figure_formats = figure_formats_,
        figure_dpi = figure_dpi,
        n_gpu_jobs = n_gpu_jobs_,
        no_gpu= no_gpu,
        # Retry parameters
        stop_max_attempt_number = stop_max_attempt_number,
        wait_random_min = stop_max_attempt_number,
        wait_random_max = stop_max_attempt_number,
        # Clustering parameters
        cluster_method = cluster_method.lower(),
        cluster_metric = cluster_metric.lower(),
        mepp_plot_format = mepp_plot_format.lower(),
        table_image_dpi = table_image_dpi,
        table_image_format = table_image_format.lower(),
        mt_method = mt_method.lower(),
        mt_alpha = mt_alpha,
        thorough_mt = thorough_mt
    )

    for k, v in results_html_filepaths.items():
        click.echo(f'HTML result table for orientation {k}: {v}')

    for k, v in clustermap_html_filepaths.items():
        click.echo(f'HTML clustermap for orientation {k}: {v}')

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
