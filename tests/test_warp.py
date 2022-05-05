from . import conftest

import pytest
import laspy
import numpy as np

import subprocess

# Maximum allowed deviations from expected data (in georeferenced units,
# meters) in output.
OUTPUT_XY_TOLERANCE = 0.01
OUTPUT_Z_TOLERANCE = 0.05 # some slack in case DVR90 should change slightly...

# We should be writing LAS/LAZ 1.4 files
EXPECTED_MAJOR_VERSION = 1
EXPECTED_MINOR_VERSION = 4

# Modern LAS 1.4 PDRFs without and with RGB color, respectively
EXPECTED_PDRF_WITHOUT_COLOR = 6
EXPECTED_PDRF_WITH_COLOR = 7

def assert_las_version_correct(actual_lasdata):
    assert actual_lasdata.header.major_version == EXPECTED_MAJOR_VERSION
    assert actual_lasdata.header.minor_version == EXPECTED_MINOR_VERSION

def assert_pdrf_equal(actual_lasdata, expected_format_id):
    assert actual_lasdata.point_format.id == expected_format_id

def assert_xyz_approx_equal(actual_lasdata, expected_lasdata):
    np.testing.assert_allclose(actual_lasdata.points.x, expected_lasdata.points.x, rtol=0, atol=OUTPUT_XY_TOLERANCE)
    np.testing.assert_allclose(actual_lasdata.points.y, expected_lasdata.points.y, rtol=0, atol=OUTPUT_XY_TOLERANCE)
    np.testing.assert_allclose(actual_lasdata.points.z, expected_lasdata.points.z, rtol=0, atol=OUTPUT_Z_TOLERANCE)

def assert_rgb_equal(actual_lasdata, expected_lasdata):
    np.testing.assert_equal(actual_lasdata.points.red, expected_lasdata.points.red)
    np.testing.assert_equal(actual_lasdata.points.green, expected_lasdata.points.green)
    np.testing.assert_equal(actual_lasdata.points.blue, expected_lasdata.points.blue)

# TODO implement SRS check?

# TODO implement extrabyte check? (use smth like np.array(lasdata.points['Pulse width']))

def test_dvrwarp(read_expected_las, tmp_path):
    output_filename = tmp_path.joinpath('output.laz')
    
    subprocess.check_call([
        'dvrwarp',
        str(conftest.PC_WITH_PULSE_WIDTH_NO_SRS_PATH),
        str(output_filename),
    ])
    
    written_lasdata = laspy.read(output_filename)
    expected_lasdata = read_expected_las

    assert_xyz_approx_equal(written_lasdata, expected_lasdata)
    assert_las_version_correct(written_lasdata)
    assert_pdrf_equal(written_lasdata, EXPECTED_PDRF_WITHOUT_COLOR)

def test_dvrwarp_colorized(read_expected_las, tmp_path):
    output_filename = tmp_path.joinpath('output_colorized.laz')

    subprocess.check_call([
        'dvrwarp',
        str(conftest.PC_WITH_PULSE_WIDTH_NO_SRS_PATH),
        str(output_filename),
        '--color-raster',
        str(conftest.COLORIZATION_RASTER_PATH),
    ])

    written_lasdata = laspy.read(output_filename)
    expected_lasdata = read_expected_las
    
    assert_xyz_approx_equal(written_lasdata, expected_lasdata)
    assert_rgb_equal(written_lasdata, expected_lasdata)
    assert_las_version_correct(written_lasdata)
    assert_pdrf_equal(written_lasdata, EXPECTED_PDRF_WITH_COLOR)
