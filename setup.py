from setuptools import setup, find_packages

setup(
    name="mlops-iris",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "flask>=2.3.3",
        "mlflow>=2.5.0",
        "joblib>=1.3.2",
    ],
)