import setuptools

setuptools.setup(
    name="rapp",
    version="0.1.0",
    author="Tomás Juan Link",
    author_email="tomaslink@gmail.com",
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
            'rapp = cli:main',
        ]
    },
    install_requires=[
        'numpy<2',
        'matplotlib<4'
        'pyserial<4'
        'scipy<2'
    ]
)
