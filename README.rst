=====================================
Motif Enrichment Positional Profiling
=====================================

Motif Enrichment Positional Profiling (MEPP) quantifies a positional profile of motif enrichment along the length of DNA sequences centered on e.g. transcription start sites or transcription factor binding motifs.

Installation
------------
To install MEPP, use pip::
    
    pip install git+https://github.com/npdeloss/mepp@main

Or, if you only have user privileges::
    
    pip install git+https://github.com/npdeloss/mepp@main --user

You may need to append the following to your ~/.bashrc::
    
    export PATH="$HOME/.local/bin:$PATH"

Usage
-----

Motif files for use with this program can be found in the data subdirectory.
These are motifs from the `HOMER <http://homer.ucsd.edu/homer/>`_ suite in data/homer.motifs.txt, as well as a reduced-redundancy version with similar motifs clustered, allowing a faster analysis. The file data/ohler_motifs.txt contains Drosophila core promoter motifs from `Ohler et al. <https://pubmed.ncbi.nlm.nih.gov/12537576/>`_. To get started, see our `walkthrough notebook <https://github.com/npdeloss/mepp/blob/main/notebooks/Walkthrough_Motif_Centered_BigWig_Scored_Analysis.ipynb>`_.

Command line help::
    
    Usage: mepp [OPTIONS]

  Profile positional enrichment of motifs in a list of scored sequences.
  Generated MEPP (Motif Enrichment Positional Profile) plots.

    Options:
      --fa TEXT                       Path to a scored fasta file, where sequence
                                      headers are of the form: ">sequence_name
                                      sequence_score".  [required]
      --motifs TEXT                   Path to a motif matrices file in JASPAR
                                      format. As a start, one can be obtained
                                      through the JASPAR website at:
                                      http://jaspar.genereg.net/downloads/
                                      [required]
      --out TEXT                      Create this directory and write output to
                                      it.  [required]
      --center INTEGER                0-based offset from the start of the
                                      sequence to center plots on. Default: Set
                                      the center to half the sequence length,
                                      rounded down
      --dgt INTEGER                   Percentage of sequence that can be
                                      degenerate (Not A, C, G, or T) before being
                                      rejected from the analysis. Useful for
                                      filtering out repeats. Default: 100
      --perms INTEGER                 Number of permutations for permutation
                                      testing and confidence intervals. Can lead
                                      to significant GPU memory usage. Default:
                                      1000
      --batch INTEGER                 Size of batches for Tensorflow datasets.
                                      Default: 1000
      --jobs INTEGER                  Number of jobs for CPU multiprocessing.
                                      Default: Use all cores
      --keepdata                      Set this flag to keep the Tensorflow dataset
                                      after MEPP has finished. Default: Delete the
                                      dataset after MEPP has finished.
      --orientations TEXT             Comma-separated list of motif orientations
                                      to analyze for CPU multiprocessing. Values
                                      in list are limited to "+" (Match motif
                                      forward orientation), "-" (Match motif
                                      reverse orientation), "+/-" (Match to
                                      forward or reverse). Default: +,+/-
      --margin INTEGER                Number of bases along either side of motif
                                      to "blur" motif matches for smoothing.
                                      Default: 2
      --pcount FLOAT                  Pseudocount for setting motif match
                                      threshold via MOODS. Default: 0.0001
      --pval FLOAT                    P-value for setting motif match threshold
                                      via MOODS. Default: 0.0001
      --bg FLOATS                     Background DNA composition, for setting
                                      motif match threshold via MOODS, represented
                                      as a series of 4 floats. Default: 0.25 0.25
                                      0.25 0.25
      --ci FLOAT                      Confidence interval for positional profile,
                                      expressed as a percentage. Default: 95.0
      --sigma FLOAT                   Adaptive scale for brightness of motif
                                      matches in motif heatmaps. Maximum
                                      brightness is achieved at sigma * std, where
                                      std is the standard deviation of nonzero
                                      motif match scores. Set lower for brighter
                                      pixels. Must be a positive value. Default:
                                      0.5
      --cmap TEXT                     Name of a matplotlib colormap. Used to color
                                      the central MEPP motif heatmap. Possible
                                      values can be viewed using
                                      matplotlib.pylot.colormaps() or at https://m
                                      atplotlib.org/stable/tutorials/colors/colorm
                                      aps.html . Default: gray_r. Set to gray to
                                      invert colors (black background).
      --smoothing INTEGER             Factor by which to smooth motif density
                                      along ranks for visualization. This is
                                      multiplicative to smoothing that already
                                      occurs dependent on figure pixel resolution.
                                      Default: 5
      --width INTEGER                 Width of generated MEPP plot, in inches.
                                      Default: 10
      --height INTEGER                Height of generated MEPP plot, in inches.
                                      Default: 10
      --formats TEXT                  Comma-separated list of image formats for
                                      MEPP plots. Possible formats are png and
                                      svg. Default: png,svg
      --dpi INTEGER                   DPI of generated MEPP plot. Default: 300
      --gjobs INTEGER                 Number of jobs for GPU multiprocessing.
                                      NOTE: Set this carefully to avoid jobs
                                      crowding each other out of GPU memory,
                                      causing profile generation to fail. If
                                      setting --nogpu, this will be the number of
                                      jobs used to process motifs in parallel.
                                      Default: 1
      --nogpu                         Disable use of GPU. If setting --nogpu,
                                      --gjobs will be the number of jobs used to
                                      process motifs in parallel.
      --attempts INTEGER              Number of attempts to retry making a plot.
                                      Default: 10
      --minwait FLOAT                 Minimum wait between attempts to make a
                                      plot, in seconds. Default: 1.0
      --maxwait FLOAT                 Maximum wait between attempts to make a
                                      plot, in seconds. Default: 1.0
      --cmethod METHOD                Clustering method for clustering MEPP
                                      profiles. For details, see "method"
                                      parameter of
                                      scipy.cluster.hierarchy.linkage. Default:
                                      average
      --cmetric METRIC                Clustering metric for clustering MEPP
                                      profiles. For details, see "metric"
                                      parameter of
                                      scipy.cluster.hierarchy.linkage. Default:
                                      correlation
      --tdpi INTEGER                  DPI of inline plots for clustering table.
                                      Default: 100
      --tformat [png|svg]             Format of inline plots for clustering table.
                                      Use png for speed, svg for publication
                                      quality. Default: png
      --mtmethod METHOD               Multiple testing method for adjusting
                                      p-values of positional correlations listed
                                      in the clustering table.For details, see
                                      "method" parameter of
                                      statsmodels.stats.multitest.multipletests.
                                      Default: fdr_by
      --mtalpha FLOAT                 Alpha (FWER, family-wise error rate) for
                                      adjusting p-values of positional
                                      correlations listed in the clustering
                                      table.For details, see "alpha" parameter of
                                      statsmodels.stats.multitest.multipletests.
                                      Default: 0.01
      --thoroughmt                    Enables thorough multiple testing of
                                      positional correlation p-values: All
                                      p-values for all motifs at all positions
                                      will be adjusted simultaneously.Default:
                                      Thorough multiple testing is enabled
      --non-thoroughmt                Disables thorough multiple testing of
                                      positional correlation p-values: Only
                                      extreme p-values will be adjusted
                                      for.Default: Thorough multiple testing is
                                      enabled
      --help                          Show this message and exit.


Motif discovery
-----
Command line help::

    Usage: python -m mepp.learn_motifs [OPTIONS]

    Options:
      --fa TEXT                       Path to a scored fasta file, where sequence
                                      headers are of the form: ">sequence_name
                                      sequence_score".  [required]
      --out TEXT                      Create this directory and write output to
                                      it.  [required]
      --dgt FLOAT                     Percentage of sequence that can be
                                      degenerate (Not A, C, G, or T) before being
                                      rejected from the analysis. Useful for
                                      filtering out repeats. Default: 100
      --batch INTEGER                 Size of batches for Tensorflow datasets.
                                      Default: 1000
      --val FLOAT                     Fraction of data used for validation.
                                      Default: 0.10
      --motifs INTEGER                Number of motifs to learn. Default: 320
      --length INTEGER                Length of motifs to learn. Default: 8
      --motif-prefix TEXT             Prefix motif names with this string.Default:
                                      denovo_motif_
      --model [deepbindlike|simpleconv]
                                      Type of network to use for learning motifs.
                                      Default: deepbindlike
      --seed INTEGER                  Random seed for shuffling and
                                      initialization. Default: 1000
      --epochs INTEGER                Maximum number of epochs for training.
                                      Default: 1000
      --no-early-stopping             Disable early stopping of training, to train
                                      for the maximum number of epochs. Default:
                                      Enable early stopping.
      --patience INTEGER              Number of epochs to wait for early stopping.
                                      Default: 1000
      --mindelta FLOAT                Minimum delta for early stopping. Default: 0
      --jobs INTEGER                  Number of jobs for CPU multiprocessing.
                                      Default: Use all cores
      --nogpu                         Disable use of GPU.
      --quiet                         Do not write combined motifs to stdout.
                                      Default: Write combined motifs to stdout.
      --help                          Show this message and exit.

Motif comparison
-----
Command line help::

    Usage: python -m mepp.compare_motifs [OPTIONS]

    Options:
      --motifs TEXT        Path to a motif matrices file in JASPAR format.
                           Preferably a denovo motif matrices file. if --known-
                           motifs is not specified, this will be compared against
                           itself. As a start, one can be obtained through the
                           JASPAR website at: http://jaspar.genereg.net/downloads/
                           [required]
      --out TEXT           Create this directory and write output to it.
                           [required]
      --known-motifs TEXT  Path to a known motif matrices file in JASPAR format.As
                           a start, one can be obtained through the JASPAR website
                           at: http://jaspar.genereg.net/downloads/ Default: None
      --overlap INTEGER    Minimum overlap for correlated motifs. Default: 6
      --corrcoef FLOAT     Minimum correlation for correlated motifs. Default: 0.6
      --combine            Combine motifs. Default: Do not combine motifs.
      --motif-prefix TEXT  Prefix motif names with this string.Default:
                           combined_motif_
      --no-logos           Do not render logos. Default: Render logos.
      --jobs INTEGER       Number of jobs for CPU multiprocessing. Default: Use
                           all cores
      --quiet              Do not write combined motifs to stdout. Default: Write
                           combined motifs to stdout.
      --help               Show this message and exit.



* Free software: MIT license

Credits
-------
- This package was developed in the `lab of Christopher Benner at UCSD <http://homer.ucsd.edu/BennerLab/>`_.
- This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
