import setuptools

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fermatrica-utils"
    , version="0.2"
    , author="OKKAM CS: AMeshkov, Ivan Kocherov, Ulyana Firyago"
    , description="FERMATRICA utils, shared objects and functions"
    , long_description=long_description
    , long_description_content_type="text/markdown"
    , url="https://github.com/FERMATRICA/fermatrica_utils"
    , packages=setuptools.find_packages(exclude=['samples*'])
    , include_package_data=True
    , classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
    , python_requires=">=3.10"
)
