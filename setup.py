import os
from setuptools import setup

from dbscan import __version__

setup(name='dbscan',
      version = ".".join(map(str, __version__)),
      packages=['dbscan']
)
