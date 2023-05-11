# Documentation Guide  #


# install
pip install mkdocs-material mkdocstrings mkdocstrings-python

# To build documentation
mkdocs build

# To deploy website locally (and rebuild when any local changes are made)
mkdocs serve

# To deploy local changes to hosted website (github pages)
mkdocs gh-deploy

# documentation project structure:
refuel-oracle/
    docs/
        api.md
        index.md
    mkdocs.yaml
    .github/
        workflows/
            docs.yaml


# MkDocs configuration
mkdocs.yaml is the top level configuration file.

In this file we specify the markdown files (found in docs directory) that we wish to include in our site

You can also specify which features, themes, and plugins to include in this yaml file


# How to use automated code documentation
The mkdocstrings plugin allows for procedurally generating documentation, example below:

::: autolabel.labeler.LabelingAgent.run
