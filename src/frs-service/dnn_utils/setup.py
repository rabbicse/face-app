import setuptools

setuptools.setup(
    name="dnn-vision_utils",
    version="1.0.1",
    author="MD. MEHEDI HASAN RABBI",
    author_email="rabbi.cse.sust.bd@gmail.com",
    description="DNN related utility package",
    packages=setuptools.find_packages(exclude=("tests",)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL-3.0",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
)
