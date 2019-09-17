from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
version_path = convert_path('statistics/_version.py')
with open(version_path, 'r') as version_file:
    exec(version_file.read(), main_ns)
version = main_ns['__version__']

setup(
    name="statistics",
    author="Pablo Romano",
    description="Statistics for Python",
    version=version,
    packages=find_packages(),
    zip_safe=False
)
