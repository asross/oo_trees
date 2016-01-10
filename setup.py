import setuptools
import io
import oo_trees

setuptools.setup(
    name='oo_trees',
    version=oo_trees.__version__,
    test_suite='tests',
    url='http://github.com/asross/oo_trees',
    license='Apache Software License',
    author='Andrew Ross',
    install_requires=[
      'numpy>=1.10.4',
      'scikit-learn>=0.17',
      'scipy>=0.16.1'
    ],
    author_email='andrewslavinross@gmail.com',
    description='Object-oriented implementations of decision tree variants',
    long_description=io.open('./README.md').read(),
    packages=['oo_trees'],
    include_package_data=True,
    platforms='any',
    classifiers=[
      'Programming Language :: Python',
      'Development Status :: Alpha',
      'Natural Language :: English',
      'License :: OSI Approved :: Apache Software License',
      'Operating System :: OS Independent',
      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research'
    ]
)
