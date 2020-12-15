from setuptools import setup


setup(name='JEOL_eds',
      description='Read binary ".pts" files',
      version='0.2',
      author='Ivo Alxneit',
      author_email='ivo.alxneit@psi.ch',
      packages=['JEOL_eds'],
      install_requires=['numpy',
                        'scipy'],
      zip_safe=False)
