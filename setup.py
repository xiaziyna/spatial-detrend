from setuptools import setup, find_packages

setup(
    name="spatial-detrend",
    version="0.1.0",
    description="A Python library to mitigate spatially-correlated systematic noise in Kepler light curves",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jamila Taaki",
    author_email="xiaziyna@gmail.com",
    url="https://github.com/xiaziyna/spatial-detrend",  
    packages=find_packages(),
    package_data={
        'spatial_detrend': ['data/*'],
    },
    install_requires=[
        "numpy",
        "scipy",
        "sklearn",
        "astropy"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",  
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

