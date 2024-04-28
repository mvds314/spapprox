from setuptools import setup

setup(
    name="spapprox",
    version="23.10",
    description="Saddle point approximation",
    url="https://github.com/mvds314/spapprox",
    author="Martin van der Schans",
    license="BSD",
    keywords="statistics",
    packages=["spapprox"],
    install_requires=["numpy", "scipy", "pandas", "statsmodels"],
    extras_require={"diff": ["numdifftools"], "fastnorm": ["fastnorm"]},
)
