# Contributing to Autolabel

Welcome, and thank you for your interest in contributing to Autolabel! We welcome contributions in all forms - bug reports, pull requests and ideas for improving the library.

1. For proposing new feature ideas or sharing your feedback, you can join our [Discord](https://discord.gg/fweVnRx6CU) to chat with the community.
2. For submitting a bug report or improvement to the documentation, you can open an [issue](https://github.com/refuel-ai/autolabel/issues) on Github.
3. For contributing code, please read the code contribution guidelines below. Open issues tagged with [`good-first-issue`](https://github.com/refuel-ai/autolabel/labels/good%20first%20issue) are good candidates if you'd like to start contributing to the repository!

## Setting Up Your Environment

1. Clone the repository:
```bash
git clone https://github.com/refuel-ai/autolabel.git
```
2. Go to the project home directory: 
```bash
cd autolabel
```
3. Install the library from source (preferably in a virtual environment): 
```bash
pip install '.[dev]'
```
4. Install [pre-commit](https://pre-commit.com/) and then run: 
```bash
pre-commit install
```
5. If the following code snippet does not return any error, the installation is successful:
```bash
python -c "from autolabel import LabelingAgent"
```

Here is a quick video that walks through the steps:

https://github.com/refuel-ai/autolabel/assets/1568137/8f63449a-6eab-4a23-bf74-1446024d86fb



## Code contribution guidelines

Please follow a ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow for submitting changes to the repository.

## Github issue guidelines

To open a new issue, go to the [issues page](https://github.com/refuel-ai/autolabel/issues).

* We have a taxonomy of labels to help organize and catalog reported issues. Please use these as much as possible to help sort issues efficiently.
* If you are adding an issue, please try to follow one of the provided templates.
