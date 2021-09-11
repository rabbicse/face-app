import os
import shutil
import setuptools

if os.path.exists('build'):
    shutil.rmtree('build')

setuptools.setup(
    name="retina-face",
    version="1.0.1",
    author="MD. MEHEDI HASAN RABBI",
    author_email="rabbi.cse.sust.bd@gmail.com",
    description="Face detection package, used retina-face algorithm.",
    package_dir={'': '..'},
    packages=setuptools.find_packages(where=os.path.abspath('..'), include=['retina_face', 'retina_face.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL-3.0",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
)
