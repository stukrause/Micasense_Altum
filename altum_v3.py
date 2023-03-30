"""
This script performs radiometric calibration on Micasense Altum images and outputs calibrated images and files.

The script aligns the images, calculates the reflectance of a reference panel, and applies the radiometric calibration to
the images in the input directory. The output is saved in the specified output directory.

The script requires a Micasense environment and the following libraries: micasense, mapboxgl, exiftool, and cv2.

"""
def main():
    # Runs with conda Micasense environment
    # Import Specific libraries
    import micasense.imageset as imageset
    import micasense.capture as capture
    import micasense.imageutils as imageutils
    import micasense.plotutils as plotutils
    from mapboxgl.utils import df_to_geojson, create_radius_stops, scale_between
    import exiftool
    import cv2

    # Import general libraries
    import os
    import glob
    import multiprocessing
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    import subprocess
    import argparse
    import datetime
    from numpy import array
    from numpy import float32
    from pathlib import Path

    # Build parser library
    parser = argparse.ArgumentParser()
    my_parser = argparse.ArgumentParser(
        description='Radiometric Calibration script for Micasense Altum images')
    parser.add_argument('-p', '--panel', required=True,
                        help='Full path to the reference panel directory. It is important that only good quality centralised panel images are used as reference')
    parser.add_argument('-i', '--imageset', required=True,
                        help='Full path to the images folder')
    parser.add_argument('-o', '--output', required=True,
                        help='Full output path where the calibrated images and files will be saved')
    parser.add_argument('-t', '--thumbnail', default=True)
    parser.add_argument('--dls', default=True)
    parser.add_argument('--overwrite', default=False)
    args = parser.parse_args()

    # Define initial settings
    useDLS = args.dls
    overwrite = args.overwrite
    generateThumbnails = args.thumbnail
    imagePath = args.imageset
    panelPath = args.panel
    outputPath = args.output + '/tif_ex'
    thumbnailPath = args.output + '/jpeg'
    start = datetime.datetime.now()

    # Create Panel Imageset
    panelset = imageset.ImageSet.from_directory(panelPath)
    panelCap = panelset.captures
    irradiances = []
    for capture in panelCap:
        if capture.panel_albedo() is not None and not any(v is None for v in capture.panel_albedo()):
            panel_reflectance_by_band = capture.panel_albedo()
            panel_irradiance = capture.panel_irradiance(
                panel_reflectance_by_band)
            irradiances.append(panel_irradiance)
        img_type = 'reflectance'
    # Get the mean reflectance per band considering all panel images
    df_panel = pd.DataFrame(irradiances)
    mean_irradiance = df_panel.mean(axis=0)
    mean_irradiance = mean_irradiance.values.tolist()

    # Load the Imageset
    imgset = imageset.ImageSet.from_directory(imagePath)
    data, columns = imgset.as_nested_lists()
    df_img = pd.DataFrame.from_records(
        data, index='timestamp', columns=columns)
    geojson_data = df_to_geojson(df_img,columns[3:], lat='latitude', lon='longitude')

    # Creating the output paths and geojson
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    if generateThumbnails and not os.path.exists(thumbnailPath):
        os.makedirs(thumbnailPath)
    #with open(os.path.join(outputPath, 'imageset.json'),'w') as f:
     #   f.write(str(geojson_data))

    # Imageset transforms
    # Alignment settings
    match_index = 1  # Index of the band I will try to match all others
    # increase max_iterations for better results, but longer runtimes
    max_alignment_iterations = 5
    warp_mode = cv2.MOTION_HOMOGRAPHY  # for Altum images only use HOMOGRAPHY
    # for images with Rigrelatives, setting this to 0 or 1 may improve the alignment
    pyramid_levels = 1

    # Find the warp_matrices for one of the images
    # chose a random middle of the flight capture (like 50?)

    matrice_sample = imgset.captures[0]
    warp_matrices, alignment_pairs = imageutils.align_capture(matrice_sample,
                                                              ref_index = match_index,
                                                              max_iterations = max_alignment_iterations,
                                                              warp_mode = warp_mode,
                                                              pyramid_levels = pyramid_levels)
    # warp_matrices = [array([[ 1.0020150e+00,  1.4219059e-03, -8.2454987e+00],
      # [ 8.0875925e-06,  1.0015880e+00, -3.5072060e+01],
      # [ 1.0681791e-06,  7.5425459e-07,  1.0000000e+00]], dtype=float32), array([[1., 0., 0.],
      # [0., 1., 0.],
      # [0., 0., 1.]], dtype=float32), array([[ 1.0011088e+00,  9.2255423e-04,  7.7074304e+00],
      # [ 1.4164108e-03,  1.0003647e+00, -2.4059565e+01],
      # [ 1.6412947e-06,  1.2169833e-06,  1.0000000e+00]], dtype=float32), array([[ 1.0010078e+00,  2.2322661e-03, -1.0162781e+00],
      # [ 6.7795871e-04,  1.0012311e+00, -4.5045223e+01],
      # [ 1.2613673e-06,  1.8045564e-06,  1.0000000e+00]], dtype=float32), array([[ 9.9970192e-01,  3.9461749e-03, -1.0079514e+01],
      # [-1.5047204e-03,  9.9975967e-01, -3.0482735e+01],
      # [ 3.0093733e-07,  2.1162466e-06,  1.0000000e+00]], dtype=float32), array([[ 6.28517187e-02, -5.65253943e-04,  1.54735415e+01],
      # [ 2.74068082e-04,  6.29043796e-02,  8.76948208e+00],
      # [-1.28265841e-06, -2.72041079e-06,  1.00000000e+00]])]
    print("Finished Aligning, warp matrices={}".format(warp_matrices))

    # save warp_matrices used for the imgset alignment
    with open(os.path.join(outputPath, 'warp_matrices.txt'), 'w') as f:
        f.write(str(warp_matrices))

    # Effectively UNWARP, ALIGN, CROP EDGES and get REFLECTANCE
    for i, capture in enumerate(imgset.captures):
        try:
            irradiance = mean_irradiance+[0]
        except NameError:
            irradiance = None

        # Create the output file names and path
        outputFilename = capture.images[0].path 
        filename = Path(outputFilename).stem + '.tif'
        thumbnailFilename = capture.images[0].path
        thumbnail = Path(outputFilename).stem +'.jpg'
        fullOutputPath = os.path.join(outputPath, filename)
        fullThumbnailPath = os.path.join(thumbnailPath, thumbnail)
        # Check the waters
        if (not os.path.exists(fullOutputPath)) or overwrite:
            if(len(capture.images) == len(imgset.captures[0].images)):
                # Unwarp and Align and get Reflectance
                capture.create_aligned_capture(
                    irradiance_list=irradiance, warp_matrices=warp_matrices)
                # Save the output images
                capture.save_capture_as_stack(fullOutputPath)
                
                if generateThumbnails:
                    capture.save_capture_as_rgb(fullThumbnailPath)

        # Clean cached data
        capture.clear_image_data()

    # Extract the metadata from the captures list and save to log.csv
    def decdeg2dms(dd):
        is_positive = dd >= 0
        dd = abs(dd)
        minutes, seconds = divmod(dd*3600, 60)
        degrees, minutes = divmod(minutes, 60)
        degrees = degrees if is_positive else -degrees
        return (degrees, minutes, seconds)
    # Build file header
    header = "SourceFile,\
    DateTimeOriginal ,\
    GPSDateStamp,GPSTimeStamp,\
    GPSLatitude,GpsLatitudeRef,\
    GPSLongitude,GPSLongitudeRef,\
    GPSAltitude,GPSAltitudeRef,\
    FocalLength,\
    XResolution,YResolution,ResolutionUnits\n"

    lines = [header]
    # get the info from each capture
    for capture in imgset.captures:
        # get lat, lon, alt and time
        outputFilename = capture.images[0].path 
        filename = Path(outputFilename).stem + '.tif'
        fullOutputPath = os.path.join(outputPath, filename)
        lat, lon, alt = capture.location()
        # write to csv in format:
        # IMG_0199_1.tif,"33 deg 32' 9.73"" N","111 deg 51' 1.41"" W",526 m Above Sea Level
        latdeg, latmin, latsec = decdeg2dms(lat)
        londeg, lonmin, lonsec = decdeg2dms(lon)
        latdir = 'North'
        if latdeg < 0:
            latdeg = -latdeg
            latdir = 'South'
        londir = 'East'
        if londeg < 0:
            londeg = -londeg
            londir = 'West'
        resolution = capture.images[0].focal_plane_resolution_px_per_mm

        linestr = '"{}",'.format(fullOutputPath)
        linestr += capture.utc_time().strftime("%Y:%m:%d %H:%M:%S,")        
        linestr += capture.utc_time().strftime("%Y:%m:%d,%H:%M:%S,")
        linestr += '"{:d} deg {:d}\' {:.2f}"" {}",{},'.format(
            int(latdeg), int(latmin), latsec, latdir[0], latdir)
        linestr += '"{:d} deg {:d}\' {:.2f}"" {}",{},{:.1f} m Above Sea Level,Above Sea Level,'.format(
            int(londeg), int(lonmin), lonsec, londir[0], londir, alt)
        linestr += '{}'.format(capture.images[0].focal_length)
        linestr += '{},{},mm'.format(resolution, resolution)
        linestr += '\n'  # when writing in text mode, the write command will convert to os.linesep
        lines.append(linestr)

    # Save the CSV with each capture metadata
    fullCsvPath = os.path.join(outputPath, 'log.csv')
    with open(fullCsvPath, 'w') as csvfile:  # create CSV
        csvfile.writelines(lines)

    # overwrite the image metadata
    if os.environ.get('exiftoolpath') is not None:
        exiftool_cmd = os.path.normpath(os.environ.get('exiftoolpath'))
    else:
        exiftool_cmd = 'exiftool'
    cmd = '{} -csv="{}" -overwrite_original "{}"'.format(exiftool_cmd, fullCsvPath, outputPath)
    subprocess.check_call(cmd)
    #subprocess.check_call(cmd, shell=True)
    print(cmd)

    end = datetime.datetime.now()
    print("Running the scrip took: {}".format(end-start))

if __name__ == "__main__":
    main()


