README – CamScanner Project
An implementation of the CamScanner app with OpenCV to convert snapshots of real world documents into clear, sharp multi-page PDF’s used to email, fax print or save to cloud.
Development Environment:
* Operation system: Win 10
* Python version: 3.7  and up
Required Libraries:
* sys
* openCV
* imutils
* numpy
How to run the program:
Run this command on the Terminal
* python Scanner.py path_input_image.jpg path_output_image.jpg
Pipeline:
* Read image
* Gray scale
* Gaussian blur
* Adaptive threshold
* Denoising
* Canny
* Find contours
* Find corners
* Transformation
* Warp
* Binarization
* Save output
Output:
output_image.jpg
