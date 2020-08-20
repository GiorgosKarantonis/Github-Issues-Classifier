from pathlib import Path
from setuptools import setup, find_packages



def get_install_requires():
    with open("requirements.txt", "r") as requirements_file:
        res = requirements_file.readlines()
        return [req.split(" ", maxsplit=1)[0].replace(";", "") for req in res if req]



setup(
    name="git-label",
    entry_points={"console_scripts": ["git-label=app:cli"]},
    long_description=Path("README.md").read_text(),
    author="Giorgos Karantonis",
    author_email="gkaranto@redhat.com, giorgos@bu.edu",
    packages=find_packages(),
    install_requires=get_install_requires(),
)
