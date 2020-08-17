#!/usr/bin/env python
import setuptools

setuptools.setup(
    name="mask_rcnn",
    version="0.0.1",
    author="Eivinas Butkus",
    author_email="eivinas.butkus@yale.edu",
    description="adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html",
    packages = ['mask_rcnn'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
