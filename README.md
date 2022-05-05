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

Test that it works by calling `pytest`:

```
pytest
```

You should now be able to use the tool from within the environment.

## Usage

When installed, the `dvrwarp` command will be available. Call `dvrwarp -h` for
help; the parameters are as follows:

```
dvrwarp [-h] [--color-raster COLOR_RASTER] [--retain-extra-dims] input_pointcloud output_pointcloud
```

| Parameter | Mandatory | Description |
| --------- | --------- | ----------- |
| `input_pointcloud` | yes | Path to input LAS/LAZ file. The SRS is assumed to EPSG:25832, regardless of the file's own SRS, in order to account for inconsistencies or incorrectly registered SRS in the delivered data (the XYZ geometry is assumed to be accepted at this point). |
| `output_pointcloud` | yes | Path of output LAS/LAZ file to write. Will have an SRS of EPSG:7416, that is, EPSG:25832 with DVR90 heights. Output will be LAS/LAZ version 1.4, and the point data record format will be 6 or 7 depending on whether colorization is enabled. |
| `--color-raster COLOR_RASTER` | no | If provided, the output pointcloud will be colorized with the raster file provided in `COLOR_RASTER`. Otherwise, a pointcloud without color attributes will be written. Note that the provided raster must contain bands for red, green and blue, and that those bands must contain 16-bit data. |
| `--retain-extra-dims` | no | If provided, include the data from extrabyte dimensions in the output data. The tool will simply forward any extrabyte dimensions as they appear in the input file, though the data type may change (e.g. a dimension containing scaled 16-bit integers may be changed into a dimension containing unscaled floating point data). |

## Examples

Basic example:

```
dvrwarp input_files/1km_NNNN_EEE.laz output_files/1km_NNNN_EEE.laz
```

Same, but colorize with raster image and include extrabyte dimensions:

```
dvrwarp input_files/1km_NNNN_EEE.laz output_files/1km_NNNN_EEE.laz orthophotos/1km_NNNN_EEE.tif --retain-extra-dims
```
