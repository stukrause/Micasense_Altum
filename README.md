# Micasense_Altum



This script performs radiometric calibration on Micasense Altum images and outputs calibrated tif stacks and thumbnails (jpegs). File naming is improved. Original script from https://github.com/mdanilevicz/multispectral_img. Workflow from https://github.com/micasense/imageprocessing.

The script aligns the images, calculates the reflectance of a reference panel, and applies the radiometric calibration to the images in the input directory. The output is saved in the specified output directory.

The script requires a Micasense environment and the following libraries: micasense, mapboxgl, exiftool, and cv2. Script may need updates depending on changes in newer versions of https://github.com/micasense/imageprocessing.

This is a command line command to run the script under Windows. \tif is originals \cal is calibration panel image \tif_ex is where to export.

python altum_v3.py -i path\tif -p path\cal -o cpath\tif_ex -t T --dls T
