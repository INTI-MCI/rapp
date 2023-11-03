import setuptools

author = "Alvarez Liliana del Valle, Etchepareborda Pablo G., Link Tomás J., Vargas Camila L."
author_email = (
    "ldalvarez@inti.gob.ar, "
    "petchepareborda@inti.gob.ar, "
    "tomaslink@gmail.com, "
    "cami.vargas24@gmail.com"
)


setuptools.setup(
    name="rapp",
    version="0.1.0",
    author=author,
    author_email=author_email,
    maintainer="Link, Tomás J.",
    description="Tools for measuring the rotation angle of the plane of polarization (RAPP).",
    long_description_content_type="text/markdown",
    url="https://github.com/INTI-metrologia/rapp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'rapp = rapp.cli:main',
        ]
    },
    install_requires=[
        'numpy<2',
        'matplotlib<4',
        'pyserial<4',
        'scipy<2'
    ]
)
