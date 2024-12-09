# File path: setup.py
"""
Package setup and distribution configuration.
"""

from setuptools import setup, find_packages

setup(
    name="journal_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.55.3",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.65.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python library for analyzing personal journals using emotional pattern detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/journal_analyzer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)