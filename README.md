# docScanner

## Description
An implementation of the CamScanner app with OpenCV to convert snapshots of real world documents into clear, sharp multi-page PDF’s used to email, fax print or save to cloud.
Development Environment:
* Operation system: Win 10
* Python version: 3.7  and up
## Required Libraries:
* sys
* openCV
* imutils
* numpy
## How to run:
Run this command on the Terminal
* python Scanner.py path_input_image.jpg path_output_image.jpg
Pipeline:
* Read image
* Gray scale
* Gaussian blur
* Canny
* Find contours
* Find corners
* Transformation
* Warp
* Binarization
* Save output
## Output:
scanned image on the same folder of scan.py named -> output_image.jpg

##Results:
<img src="https://github.com/gilad4591/docScanner/blob/master/output.jpg", width="250", height="250"/>
