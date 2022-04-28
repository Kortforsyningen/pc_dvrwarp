# pc_dvrwarp
Tool to produce DVR90 pointclouds from ellipsoidal-height pointclouds for the
Danish Elevation Model (DHM/DK-DEM).

This tool is intended to provide a well-tested, robust and simple-to-use way
of producing publishable DHM pointclouds from internal master data. This
includes warping from ellipsoidal heights to DVR90 heights, as well as
optional colorization and inclusion of extrabyte dimensions.

While the tool itself is fairly simple in implementation, variations in data
formats and behavior of underlying software may cause the intended commands to
fail. This tool is intended to shield the user from having to debug such
issues.

## Installation
Create a suitable conda environment with the following command (replace
`PC_DVRWARP_ENV` with your preferred environment name):

```
conda env create -f environment.yml -n PC_DVRWARP_ENV
```

Activate the environment, then perform an editable install of the tool by
calling the following command from the repository's root directory:

```
pip install -e .
```

You should now be able to use the tool from within the environment.

## Usage
See output of `dvrwarp -h`.
