from setuptools import find_packages, setup

VERSION = {}  # type: ignore
with open("evolutionpy/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="evolutionpy",
    packages=find_packages(),
    version=VERSION["VERSION"],
    entry_points={
        "console_scripts": [
            "evo-hello=evolutionpy.evolutionpy:main",
        ]
    },
    python_requires=">=3.8.10",
)
