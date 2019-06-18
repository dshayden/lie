from setuptools import setup


def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='lie',
  version='0.1',
  description='Lie group and algebra handling',
  long_description=readme(),
  classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7',
  ],
  keywords='lie group algebra',
  url='',
  author='David S. Hayden',
  author_email='dshayden@mit.edu',
  license='MIT',
  packages=['lie', 'lie.so2', 'lie.se2', 'lie.so3', 'lie.se3'],
  install_requires=[
    'numpy', 'scipy', 'matplotlib',
  ],
  test_suite='nose.collector',
  tests_require=['nose', 'nose-cover3'],
  include_package_data=True,
  zip_safe=False)

