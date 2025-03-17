from setuptools import setup, find_packages

setup(
    name='synapedge',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'onnx>=1.10.1',

    ],
    entry_points={
        'console_scripts': [

        ],
    },
    author='Asad Shafi',
    author_email='asadshafi5@gmail.com',
    description='A brief description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/asad-shafi/synapedge',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)