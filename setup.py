from setuptools import setup, find_packages
import io
import os

setup(
    name="HyRL",
    version="1.0.0",
    description="Hybrid Reinforcement Learning for Obstacle Avoidance",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="HSL-UCSC",
    url="https://github.com/HSL-UCSC/ObstacleAvoidanceHyRL",
    # Automatically find packages, but exclude 'examples' and 'tests' directories.
    packages=find_packages(where="src", exclude=["tests*", "examples*"]),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        # key = the package name (folder under src/).
        # Here 'models' means src/models
        "HyRL": ["models/*.zip"],  # all files in models/
    },
    # package_data={
    #     # This tells setuptools to include all files under the "models" subdirectory
    #     # of the HyRL package.
    #     "HyRL": ["models/*"],
    # },
    install_requires=[
        "gym",
        "torch",
        "scikit-learn",
        "numpy",
        "stable-baselines3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
