from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
version_path = convert_path('stats/_version.py')
with open(version_path, 'r') as version_file:
    exec(version_file.read(), main_ns)
version = main_ns['__version__']

setup(
    name="stats",
    author="Pablo Romano",
    description="Statistics for Python",
    version=version,
    packages=find_packages(),
    zip_safe=False
)
