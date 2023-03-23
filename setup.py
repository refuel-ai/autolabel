import os

from setuptools import find_packages, setup

current_file_path = here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as requirements:
    install_requires = requirements.read().split("\n")

with open(os.path.join(current_file_path, "README.md"), encoding="utf-8") as rd:
    long_description = "\n" + rd.read()

setup(
    name="refuel_oracle",
    version="0.0.0",
    maintainer="Refuel.ai",
    author="Refuel.ai",
    maintainer_email="support@refuel.ai",
    author_email="support@refuel.ai",
    packages=find_packages(),
    description="Library to label your NLP datasets using Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
