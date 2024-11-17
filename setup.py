from setuptools import find_packages, setup

setup(
    name="pausenormeffect",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "librosa",
        "jaconv",
        "pyopenjtalk",
        "tqdm",
        "pyyaml",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "mypy",
        ]
    },
)
