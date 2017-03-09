import tensorflow as tf
import numpy as np
image_size = 300.0
layer_boxes = [12, 12, 12, 12, 12, 12] 
# layer_boxes = [6,6,6,6,6,6] 
classes = 1
box_ratios = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
box_s_min = 0.1
negposratio = 3

# to be set programmatically
out_shapes = None
defaults = None
