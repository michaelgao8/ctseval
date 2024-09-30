from setuptools import setup, Extension, find_packages
import numpy as np

# Define the extension module
ctseval_module = Extension(
    'ctseval._ctseval',
    sources=[
        'ctseval/ctseval.c',
        'ctseval/_ctseval.c',
    ],
    include_dirs=[np.get_include(), 'ctseval/include'],
)

# Setup the package
setup(
    name='ctseval',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[ctseval_module],
    package_data={
        'ctseval': ['include/*.h'],
    },
    include_package_data=True,
    author='Michael Gao',
    author_email='michaelgao8@gmail.com',
    description='A package for evaluating clinical time series predictions',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/michaelgao8/ctseval',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
    ],
)