import argparse
import subprocess

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('input_pointcloud', type=str, help='path to input pointcloud file')
    argument_parser.add_argument('output_pointcloud', type=str, help='path to desired output pointcloud file')
    argument_parser.add_argument('--color-raster', type=str, help='path to colorization raster file')
    argument_parser.add_argument('--retain-extra-dims', action='store_const', const=True)

    input_arguments = argument_parser.parse_args()

    input_filename = input_arguments.input_pointcloud
    output_filename = input_arguments.output_pointcloud
    
    pdal_args = [
        "pdal",
        "translate",
        input_filename,
        output_filename,
    ]

    pdal_args += [
        "assign",
        "--filters.assign.assignment=Classification[32:255]=1",
    ]

    if input_arguments.color_raster is not None:
        pdal_args += [
            "colorization",
            f"--filters.colorization.raster={input_arguments.color_raster}"
        ]
        output_dataformat = 7 # modern PDRF with RGB color
    else:
        output_dataformat = 6 # modern PDRF without color
    
    pdal_args += [
        "reprojection",
        "--readers.las.nosrs=true", # don't throw error with wrong/missing SRS
        "--readers.las.override_srs=EPSG:25832",
        "--filters.reprojection.in_srs=EPSG:25832",
        "--filters.reprojection.out_srs=EPSG:7416",
        "--writers.las.minor_version=4",
        f"--writers.las.dataformat_id={output_dataformat}",
    ]

    if input_arguments.retain_extra_dims:
        pdal_args += ["--writers.las.extra_dims=all"]

    subprocess.check_call(pdal_args)
    