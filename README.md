# Repo under construction

This repo has been recently added on Github, so it lacks clear explanation and concise code for now. The repo's changes will be completed over time. Thanks for your understanding.

# Indoor/Outdoor Positioning and Navigation

Indoor/Outdoor Positioning and Navigation is the name of my diploma thesis, which I conducted in school of Rural and Surveying Engineering at Aristotle University of Thessaloniki. 

The aim of this thesis is the development of an application by which moving objects are monitored using off the shelf video cameras and an alert signal is sent when these objects cross a beforehand defined “restricted” area. This could be used in a factory premises where workers are moving around and in case when somebody approaches “restricted” areas, e.g. machinery, and for security reasons should be kept away.

This application is written in Python programming language using mainly the
OpenCV library.

Two video cameras are located apart on the wall of an indoor area to register the objects in it. From the overlapping footage of the two video cameras, consecutive images are isolated and after processing, 3D bounding boxes are computed around the moving objects. When the moving objects enter in the already calculated “restricted area”, an alarm signal is issued.

The first step in the application is to calculate the internal and external orientation parameters and correct the lens distortions from the overlapping images. 

Two methods are used for the creation of bounding boxes around the moving objects. A change detection method based on the removal of the fixed background of the image where the moving objects appear and 2D bounding boxes are calculated around them and an object detection method where a neural network already trained to track moving objects in images and places 2D bounding boxes around them.

Based on these methods and the 3D information from the external orientation parameters of the images of the space under surveillance, 3D bounding boxes are determined around moving objects and the “restricted areas” are determined and both appear on the footage.

Finally, an alert signal is created every time a moving object crosses a “restricted” area.

## Roadmap

Due to the limited time handing in the diploma thesis, a lot of changes and optimizations need to be done for the sake of clarification, the speed and the easy usage of the application. Below, there is a list with the changes that will be applied in the future:

- [ ] Change filenames to be more precise
- [ ] Put scripts in order
- [ ] Use terminal arguments instead of absolute paths inside scripts
- [ ] Change variables' and functions' name to be more precise
- [ ] Make code more Pythonic
- [ ] Optimize code in terms of time
- [ ] Documenting Python code
- [ ] Make the project a Python module
- [ ] Create a GUI

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. For any questions, please contact with me on my email g.karantaidis@protonmail.com.

## Mandatory external files

In order for the application to run, two extra files need to be downloaded, the first from [here](https://drive.google.com/file/d/18c39RFxMgVj2VSO0KbvLrS57ztJDssy1/view?usp=sharing, https://drive.google.com/file/d/1HkRmBy8Rkd2vN-6FgmuDRf0BrKRWH2rK/view?usp=sharing) and the second from [here](https://drive.google.com/file/d/18c39RFxMgVj2VSO0KbvLrS57ztJDssy1/view?usp=sharing). Both of them should be put on weights folder.

