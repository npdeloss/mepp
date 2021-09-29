=====================================
Motif Enrichment Positional Profiling
=====================================


.. image:: https://img.shields.io/pypi/v/mepp.svg
        :target: https://pypi.python.org/pypi/mepp

.. image:: https://img.shields.io/travis/npdeloss/mepp.svg
        :target: https://travis-ci.com/npdeloss/mepp

.. image:: https://readthedocs.org/projects/mepp/badge/?version=latest
        :target: https://mepp.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/npdeloss/mepp/shield.svg
     :target: https://pyup.io/repos/github/npdeloss/mepp/
     :alt: Updates



Motif Enrichment Positional Profiling (MEPP) quantifies a positional profile of motif enrichment along the length of DNA sequences centered on e.g. transcription start sites or transcription factor binding motifs.

Command line help::
    Usage: mepp [OPTIONS]

      Profile positional enrichment of motifs in a list of scored sequences.
      Generated MEPP (Motif Enrichment Positional Profile) plots.

    Options:
      --fa TEXT                       Path to a scored fasta file, where sequence
                                      headers are of the form: ">sequence_name
                                      sequence_score".  [required]
      --motifs TEXT                   Path to a motif matrices file in JASPAR
                                      format.As a start, one can be obtained
                                      through the JASPAR website at:
                                      http://jaspar.genereg.net/downloads/
                                      [required]
      --out TEXT                      Create this directory and write output to
                                      it.  [required]
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
      --bg FLOAT...                   Background DNA composition, for setting
                                      motif match threshold via MOODS, represented
                                      as a series of 4 floats. Default: 0.25 0.25
                                      0.25 0.25
      --sigma FLOAT                   Adaptive scale for brightness of motif
                                      matches in motif heatmaps. Maximum
                                      brightness is achieved at sigma * std, where
                                      std is the standard deviation of nonzero
                                      motif match scores. Set lower for brighter
                                      pixels. Must be a positive value. Default:
                                      0.5
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
                                      causing profile generation to fail. Default:
                                      1
      --cmethod [average|single|complete|centroid|median|ward|weighted]
                                      Clustering method for clustering MEPP
                                      profiles. For details, see "method"
                                      parameter of
                                      scipy.cluster.hierarchy.linkage. Default:
                                      average
      --cmetric [correlation|euclidean|minkowski|cityblock|seuclidean|sqeuclidean|cosine|jensenshannon|chebyshev|canberra|braycurtis|mahalanobis]
                                      Clustering metric for clustering MEPP
                                      profiles. For details, see "metric"
                                      parameter of
                                      scipy.cluster.hierarchy.linkage. Default:
                                      correlation
      --tdpi INTEGER                  DPI of inline plots for clustering table.
                                      Default: 100
      --tformat [png|svg]             Format of inline plots for clustering table.
                                      Use png for speed, svg for publication
                                      quality. Default: png
      --mtmethod [fdr_tsbky|bonferroni|sidak|holm-sidak|holm|simes-hochberg|hommel|fdr_bh|fdr_by|fdr_tsbh]
                                      Multiple testing method for adjusting
                                      p-values of positional correlations listed
                                      in the clustering table.For details, see
                                      "method" parameter of
                                      statsmodels.stats.multitest.multipletests.
                                      Default: fdr_tsbky
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

* Free software: MIT license
* Documentation: https://mepp.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
