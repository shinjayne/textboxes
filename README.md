#tb_tensorflow
A python implementation of TextBoxes

## Dependencies

* TensorFlow r1.0
* OpenCV2

Code from Chaoyue Wang</br>

03/09/2017 Update:</br>

1.Debugging optimizer function in trainer.py, still with same issue.</br>
2.Fixed several bugs of match box computation.</br>

03/05/2017 Update:</br>

1.Changed and added image processing function to pre and post process input and output data.</br>
2.Tested training, but optimizer initialization function has segmentation fault issue.</br>

03/03/2017 Update:</br>

1.Added svt_data_loader.py to parse svt dataset xml config file.</br>
2.Modified trainer.py to adapt textboxes training and test.</br>

03/01/2017 Update:</br>

1.Revised default_boxes function to use numpy ndarray. Because TextBoxes need more default boxes than SSD, python list cannot handle them.</br>
2.Adjusted the architecture of directory.</br>

02/25/2017 Update:</br>

1.Revised convolution layers setting.</br>
2.Removed useless functions(webcam support).</br>

02/22/2017 Update:</br>

1.Forked reference code from seann999/ssd_tensorflow.</br>