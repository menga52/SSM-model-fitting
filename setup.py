from setuptools import setup, find_packages

setup(
    name="Playground",  # Name of your module
    version="0.1",     # Module version
    packages=find_packages(),  # Automatically finds all sub-packages
    install_requires=['numpy>=2.2.3','jax>=0.4.37'],  # dependencies
  )
