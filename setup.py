#!/usr/bin/env python
#-*- coding:utf-8 -*-
 
#############################################
# File Name: setup.py
# Author: Xiaoyang Chen
# Mail: xychen20@mails.tsinghua.edu.cn
# Created Time: Sat 28 Nov 2020 08:31:29 PM CST
#############################################


from setuptools import setup, find_packages

setup(
    name="stPlus",
    version="0.0.6",
    keywords=("pip", "stPlus"),
    description="stPlus: reference-based enhancement of spatial transcriptomics",
    long_description="stPlus is a reference-based method for the enhancement of spatial transcriptomics. Leveraging the holistic information in reference scRNA-seq data but not limited to the genes shared with spatial data, stPlus performs non-linear embedding for cells in both datasets and effectively predicts unmeasured spatial gene expression. All stPlus wheels distributed on PyPI are MIT licensed.",
    license="MIT Licence",
    url="https://github.com/xy-chen16/stPlus",
    author="Xiaoyang Chen",
    author_email="xychen20@mails.tsinghua.edu.cn",
    maintainer = "Shengquan Chen",
    maintainer_email = "ccq17@mails.tsinghua.edu.cn",
    packages=find_packages(),
    python_requires='>3.6.0',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    install_requires=[
        'torch>=1.6.0',
        'torchvision>=0.7.0',
        'pandas>=1.1.0b',
        'numpy>=1.19.1',
        'scipy>=1.5.2',
        'scikit-learn>=0.23.2'
    ]
)

