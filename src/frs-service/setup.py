import setuptools

setuptools.setup(
    name="frs-service",
    version="1.0.1",
    author="MD. MEHEDI HASAN RABBI",
    author_email="rabbi.cse.sust.bd@gmail.com",
    description="Face embedding generator and match embeddings. Used arcface algorithm",
    packages=setuptools.find_packages("arc_face"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL-3.0",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
)
