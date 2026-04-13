from setuptools import setup


setup(name='JEOL_eds',
      description='Read binary files acquired by JEOL Analysis Station',
      version='2.1',
      author='Ivo Alxneit',
      author_email='ivo.alxneit@psi.ch',
      packages=['JEOL_eds'],
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'scikit-image',
                        'h5py',
                        'numba'],
      zip_safe=False)
