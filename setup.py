import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dea-py",
    version="0.0.1",
    author="firat tamur",
    author_email="ftamur16@ku.edu.tr",
    description="A Python Package for Data Envelopment Analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/firattamur/dea-py",
    project_urls={
        "Bug Tracker": "https://github.com/firattamur/dea-py/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "dea"},
    packages=setuptools.find_packages(where="dea"),
    python_requires=">=3.8",
)