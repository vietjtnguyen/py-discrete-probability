from distutils.core import setup

import discrete_probability

setup(
    name='pyDiscreteProbability',
    version=discrete_probability.__version__,
    author='Viet Nguyen',
    author_email='vnguyen@cs.ucla.edu',
    packages=['discrete_probability'],
    scripts=[],
    url='http://pypi.python.org/pypi/pyDiscreteProbability/',
    license='LICENSE.txt',
    description='Discrete probability learning and inference in Python.',
    long_description=open('README.txt').read(),
    install_requires=[
    ],
    classifiers=[
		'Topic :: Education',
		'Topic :: Scientific/Engineering :: Artificial Intelligence',
		'Topic :: Scientific/Engineering :: Information Analysis',
		'Topic :: Scientific/Engineering :: Mathematics',
		'Intended Audience :: Education',
		'Intended Audience :: Science/Research',
		'Natural Language :: English',
		'Programming Language :: Python',
		],
)

