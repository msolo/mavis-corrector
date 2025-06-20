from setuptools import setup

setup(
    name="mavis-corrector",
    version="0.1.2",
    packages=[
        "mavis",
        "mavis.app",
        "mavis.corrector",
        "mavis.stc",
    ],
    package_data={"mavis.stc": ["en-words.txt", "en.json"]},
)
