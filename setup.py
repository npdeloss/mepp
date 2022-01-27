#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'biopython>=1.76',
    'Click>=8.0.1',
    'joblib>=1.0.1',
    'logomaker>=0.8',
    'matplotlib>=3.3.4',
    'MOODS-python>=1.9.4',
    'numpy>=1.19',
    'pandas>=1.1.5',
    'jinja2>=2.11.3',
    'PyYAML>=5.4.1',
    'scikit-image>=0.17.2',
    'scipy>=1.5.3',
    'slugify>=0.0.1',
    'statsmodels>=0.12.2',
    'tensorflow>=2.5',
    'tqdm>=4.61.2'
]

test_requirements = ['pytest>=3', ]

setup(
    author="Nathaniel Delos Santos",
    author_email='Nathaniel.P.DelosSantos@jacobs.ucsd.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Motif Enrichment Positional Profiling (MEPP) quantifies a positional profile of motif enrichment along the length of DNA sequences centered on e.g. transcription start sites or transcription factor binding motifs.",
    entry_points={
        'console_scripts': [
            'mepp=mepp.cli:main',
            'mepp_run_single=mepp.single:run_single',
            'getscoredfasta=mepp.get_scored_fasta:main'
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mepp',
    name='mepp',
    packages=find_packages(include=['mepp', 'mepp.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/npdeloss/mepp',
    version='0.0.1',
    zip_safe=False,
)
