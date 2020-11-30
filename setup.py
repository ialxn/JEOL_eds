from setuptools import setup


setup(name='JEOL_eds',
      description='Read binary ".pts" files',
      version='0.4',
      author='Ivo Alxneit',
      author_email='ivo.alxneit@psi.ch',
      packages=['JEOL_eds'],
      install_requires=['numpy'],
      zip_safe=False)
