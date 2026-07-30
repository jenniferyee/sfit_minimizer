from setuptools import find_packages, setup

setup(
    name="sfit_minimizer",
    package_dir={"": "source"},
    packages=find_packages(where="source"),
)
