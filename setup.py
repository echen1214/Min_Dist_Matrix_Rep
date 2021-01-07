#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
                'alphaspace2==0.1.2',
                'biopython==1.78',
                'numpy>=1.19.4',
                'ProDy>=1.11',
                'scipy==1.5.4',
                'requests==2.25.0',
                'pytest==6.1.2',
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Eric Anthony Chen",
    author_email='eac709@nyu.edu',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Calculates distance matrices for a set of structures and run analyses that classify the conformational ensemble and identifies important interactions",
    entry_points={
        'console_scripts': [
            'dist_analy=dist_analy.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='dist_analy',
    name='dist_analy',
    packages=find_packages(include=['dist_analy', 'dist_analy.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/echen1214/dist_analy',
    version='0.1.0',
    zip_safe=False,
)
