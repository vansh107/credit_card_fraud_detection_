import subprocess

from setuptools import Command, find_packages, setup


class DVCSetup(Command):
    """Run DVC commands after package installation."""

    description = "Initialize and configure DVC"
    user_options = []

    def run(self):
        subprocess.check_call(["dvc", "init"])
        subprocess.check_call(["dvc", "dag"])
        print("DVC setup complete now execute `dvc repro`")


# Read requirements.txt
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("-e")]

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="This Repo is dedicated to end-to-end Machine Learning Project with MLOps",
    author="raj-maharajwala",
    license="MIT",
    install_requires=requirements,
    cmdclass={"dvc_setup": DVCSetup},
    python_requires=">=3.8",  # Specify your minimum Python version
)
