# Face_Parsing_2016

See Face_Parsing_2016/matlab/face_parsing/run_demo.m for details. Please download the data_db from [MultiHelen_v1.mat](https://www.dropbox.com/s/jtkqbfa7x0h08hn/MultiHelen_v1.mat?dl=0).

Update:

For original dataset, please find the download link here: [HelenDataset](http://pages.cs.wisc.edu/~lizhang/projects/face-parsing/).

Per request, I am adding a similar python version [HERE](https://github.com/Liusifei/Face_parsing_python.git). The python version should be compatible with many newer caffe versions (one caffe_dev is included). All parameters are the same except a slightly different face cropping strategy. Note that the python version uses IOU instead of f-score in the test.py, you need to modify it to compare with SOTA methods.
