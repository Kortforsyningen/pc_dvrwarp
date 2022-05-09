from . import conftest

import pytest
import laspy
from laspy.vlrs.known import LasZipVlr, WktCoordinateSystemVlr
from osgeo import osr
import numpy as np

import subprocess
import json

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

def assert_is_laz(filename, expected_is_laz):
    laszip_user_id = LasZipVlr.official_user_id()
    
    # A bit hackish - Laspy seems to hide the presence of LASzip compression from the user, so inspect with PDAL instead
    pdal_info_output = subprocess.check_output(['pdal', 'info', '--metadata', str(filename)])
    info_obj = json.loads(pdal_info_output)
    metadata = info_obj['metadata']
    vlrs = [metadata[key] for key in metadata if key.startswith('vlr_')]
    vlr_user_ids = [vlr['user_id'] for vlr in vlrs]

    has_laszip_vlr = laszip_user_id in vlr_user_ids
    assert has_laszip_vlr == expected_is_laz

def assert_srs_correct(actual_lasdata):
    expected_srs = osr.SpatialReference()
    expected_srs.ImportFromEPSG(7416)
    
    coordinate_system_user_id = WktCoordinateSystemVlr.official_user_id()
    
    actual_wkt_vlr = actual_lasdata.vlrs.get_by_id(user_id=coordinate_system_user_id)[0]
    actual_srs = osr.SpatialReference()
    actual_srs.ImportFromWkt(actual_wkt_vlr.string)

    assert actual_srs.IsSame(expected_srs)

# TODO implement extrabyte check? (use smth like np.array(lasdata.points['Pulse width']))

@pytest.mark.parametrize("input_path", conftest.INPUT_PC_PATHS)
@pytest.mark.parametrize("write_compressed", [False, True])
def test_dvrwarp(input_path, write_compressed, read_expected_las, tmp_path):
    if write_compressed:
        output_extension = 'laz'
    else:
        output_extension = 'las'
    output_filename = tmp_path.joinpath(f'output.{output_extension}')
    
    subprocess.check_call([
        'dvrwarp',
        str(input_path),
        str(output_filename),
    ])
    
    written_lasdata = laspy.read(output_filename)
    expected_lasdata = read_expected_las

    assert_xyz_approx_equal(written_lasdata, expected_lasdata)
    assert_srs_correct(written_lasdata)
    assert_las_version_correct(written_lasdata)
    assert_pdrf_equal(written_lasdata, EXPECTED_PDRF_WITHOUT_COLOR)
    assert_is_laz(output_filename, write_compressed)

@pytest.mark.parametrize("input_path", conftest.INPUT_PC_PATHS)
@pytest.mark.parametrize("write_compressed", [False, True])
def test_dvrwarp_colorized(input_path, write_compressed, read_expected_las, tmp_path):
    if write_compressed:
        output_extension = 'laz'
    else:
        output_extension = 'las'
    output_filename = tmp_path.joinpath(f'output_colorized.{output_extension}')

    subprocess.check_call([
        'dvrwarp',
        str(input_path),
        str(output_filename),
        '--color-raster',
        str(conftest.COLORIZATION_RASTER_PATH),
    ])

    written_lasdata = laspy.read(output_filename)
    expected_lasdata = read_expected_las
    
    assert_xyz_approx_equal(written_lasdata, expected_lasdata)
    assert_rgb_equal(written_lasdata, expected_lasdata)
    assert_srs_correct(written_lasdata)
    assert_las_version_correct(written_lasdata)
    assert_pdrf_equal(written_lasdata, EXPECTED_PDRF_WITH_COLOR)
    assert_is_laz(output_filename, write_compressed)
