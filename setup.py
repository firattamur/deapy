import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deapy",
    version="0.0.1",
    author="firat tamur",
    author_email="ftamur16@ku.edu.tr",
    description="A Python Package for Data Envelopment Analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/firattamur/deapy",
    project_urls={
        "Bug Tracker": "https://github.com/firattamur/deapy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "deapy"},
    packages=setuptools.find_packages(where="deapy"),
    python_requires=">=3.8",
)