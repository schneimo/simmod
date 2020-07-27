from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

extras = {
  'mujoco': ['mujoco_py>=2.0', 'imageio'],
    #'bullet': ['pybullet>=1.7.8']
}

# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(
    name="simmod",  # Replace with your own username
    version="0.1dev",
    author="Moritz Schneider",
    author_email="moritz.schneider1@rwth-aachen.de",
    description="A framework to change parameters of common physics simulations on-the-fly without reloading.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy>=1.18.5',
        'gym>=0.17.1',
    ],
    extras_require=extras,
    url="https://gitlab.is.tue.mpg.de/mschneider/simulation-modification-framework",
    packages=[package for package in find_packages() if package.startswith('simmod')],
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.8',
    ],
)
