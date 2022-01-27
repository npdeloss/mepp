"""Top-level package for Motif Enrichment Positional Profiling."""

__author__ = """Nathaniel Delos Santos"""
__email__ = 'Nathaniel.P.DelosSantos@jacobs.ucsd.edu'
__version__ = '0.1.0'

import sys

from .cli import main
from .get_scored_fasta import main as getfasta

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
