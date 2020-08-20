# Github-Issues-Classifier
# Copyright(C) 2020 Georgios (Giorgos) Karantonis
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

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
