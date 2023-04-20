import os

from setuptools import find_packages, setup

current_file_path = here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as requirements:
    install_requires = requirements.read().split("\n")

with open(os.path.join(current_file_path, "README.md"), encoding="utf-8") as rd:
    long_description = "\n" + rd.read()


def get_extra_requires(path, add_all=True):
    import re
    from collections import defaultdict

    tasks = [
        "multi_choice_question_answering",
        "classification",
        "named_entity_recognition",
    ]
    providers = ["openai", "openai_chat", "anthropic", "cohere", "huggingface"]

    with open(path) as file:
        extra_deps = defaultdict(set)
        for row in file:
            if row.strip() and not row.startswith("#"):
                if ":" in row:
                    dep, settings = row.split(":")
                    for setting in settings.split(","):
                        setting = setting.strip()
                        if setting in tasks:
                            for provider in providers:
                                extra_deps[setting + "_" + provider].add(dep)
                        elif setting in providers:
                            for task in tasks:
                                extra_deps[task + "_" + setting].add(dep)
                # add individual dependency options
                extra_deps[re.split("[<=>]", dep)[0]].add(dep)

        # add all extra dependencies option
        if add_all:
            extra_deps["all"] = set(dep for deps in extra_deps.values() for dep in deps)

    return extra_deps


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
    extras_require=get_extra_requires("extra-requirements.txt"),
)
