from setuptools import setup

setup(
    name="hyrl",
    version="1.0.0",
    description="Hybrid Reinforcement Learning for Obstacle Avoidance",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="HSL-UCSC",
    url="https://github.com/HSL-UCSC/ObstacleAvoidanceHyRL",
    # If modules are in the repository root, list them here:
    py_modules=["HyRL", "obstacleavoidance_env", "utils"],
    install_requires=[
        "gym",
        "torch",
        "scikit-learn",
        "numpy",
        "stable-baselines3",
    ],
    classifiers=[],
)
