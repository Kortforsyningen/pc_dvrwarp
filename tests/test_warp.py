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

EXPECTED_SRS_EPSG = 7416
EXPECTED_SRS = osr.SpatialReference()
EXPECTED_SRS.ImportFromEPSG(EXPECTED_SRS_EPSG)

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
    coordinate_system_user_id = WktCoordinateSystemVlr.official_user_id()
    
    actual_wkt_vlr = actual_lasdata.vlrs.get_by_id(user_id=coordinate_system_user_id)[0]
    actual_srs = osr.SpatialReference()
    actual_srs.ImportFromWkt(actual_wkt_vlr.string)

    assert actual_srs.IsSame(EXPECTED_SRS)

def assert_extradims_approx_equal(actual_lasdata, expected_lasdata):
    expected_extradim_names = expected_lasdata.point_format.extra_dimension_names

    for extradim_name in expected_extradim_names:
        expected_dimension = expected_lasdata.point_format.dimension_by_name(extradim_name)
        actual_dimension = actual_lasdata.point_format.dimension_by_name(extradim_name)

        # We can't currently count on description to be correct in PDAL's output :-(
        # assert actual_dimension.description == expected_dimension.description

        # Check that the values in the dimension are preserved (we currently
        # tolerate different datatype and scale/offset, as long as the
        # effective output values are preserved to within rounding errors.)
        expected_values = np.array(expected_lasdata.points[extradim_name])
        actual_values = np.array(actual_lasdata.points[extradim_name])
        np.testing.assert_allclose(actual_values, expected_values)

@pytest.mark.parametrize("input_path", conftest.INPUT_PC_PATHS)
@pytest.mark.parametrize("colorization_raster", [None, conftest.COLORIZATION_RASTER_PATH])
@pytest.mark.parametrize("retain_extra_dims", [False, True])
@pytest.mark.parametrize("write_compressed", [False, True])
def test_dvrwarp(input_path, colorization_raster, retain_extra_dims, write_compressed, read_expected_las, tmp_path):
    if write_compressed:
        output_extension = 'laz'
    else:
        output_extension = 'las'
    output_path = tmp_path.joinpath(f'output.{output_extension}')
    
    call_args = [
        'dvrwarp',
        str(input_path),
        str(output_path),
    ]
    if colorization_raster is not None:
        call_args += [
            '--color-raster',
            str(colorization_raster),
        ]
    if retain_extra_dims:
        call_args += ['--retain-extra-dims']

    # Print resulting argument list for easier debugging (will be captured by
    # pytest unless it is called with the -s flag)
    print('call:\n{}'.format(' '.join(call_args)))
    subprocess.check_call(call_args)
    
    input_lasdata = laspy.read(input_path)
    written_lasdata = laspy.read(output_path)
    expected_lasdata = read_expected_las

    assert_xyz_approx_equal(written_lasdata, expected_lasdata)
    assert_srs_correct(written_lasdata)
    assert_las_version_correct(written_lasdata)
    
    if colorization_raster is None:
        assert_pdrf_equal(written_lasdata, EXPECTED_PDRF_WITHOUT_COLOR)
    else:
        assert_pdrf_equal(written_lasdata, EXPECTED_PDRF_WITH_COLOR)
        assert_rgb_equal(written_lasdata, expected_lasdata)
    
    if retain_extra_dims:
        # The "expected" values for the extrabyte dimensions are a verbatim
        # carryover from the input data
        assert_extradims_approx_equal(written_lasdata, input_lasdata)

    assert_is_laz(output_path, write_compressed)
