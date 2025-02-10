from setuptools import setup


setup(
      name='hydromass',    # This is the name of your PyPI-package.
      version='0.7.0',
      description='Hydrostatic mass calculator',
      author='Dominique Eckert, Vittorio Ghirardini, Stefano Ettori',
      author_email='Dominique.Eckert@unige.ch',
      url="https://github.com/domeckert/hydromass",
      packages=['hydromass'],
      install_requires=[
            'numpy','scipy','astropy','matplotlib','pymc','pytensor','pyproffit','numpyro','corner'
      ],
      include_package_data=True,
)

