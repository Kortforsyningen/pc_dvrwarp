from setuptools import setup

setup(
    name="pc_dvrwarp",
    description="Tool to produce DVR90 pointclouds",
    license="MIT",
    author="Danish Agency for Data Supply and Efficiency (SDFE)",
    author_email="sdfe@sdfe.dk",
    entry_points={
        "console_scripts": [
            "dvrwarp = dvrwarp.dvrwarp:main",
        ],
    },
)
