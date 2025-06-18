from setuptools import find_packages, setup

# read requirement.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="retarget",
    version="0.0.1",
    author="RPL",
    license="MIT",
    packages=find_packages(include=[f"retarget.*"]),
    package_data={
        "configs": ["configs/*.yaml"],
    },
    install_requires=required,
    extras_require={},
)
