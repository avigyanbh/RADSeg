# Visualizers
This directory includes all visualizer classes.
Currently we only support ReRun or publishing to ROS2 topica but a layer of abstraction was created
such that we can easily support other visualizers. By simply implementing a few
core methods like (log_pc, log_img...etc.) we can add more.

The team is interested in adding open3d, and foxglove as other options.
However no immediate plans to do that yet. If the reader is interested, let us
know and we can work with you on a PR.

## Adding a visualizer
0. Read the [CONTRIBUTING](../../CONTRIBUTING.md) file.
1. Create a new python file with the same name as your visualizer.
2. Extend the base abstract class found in [base.py](base.py).
3. Implement and override the inherited methods.
4. Add a config file with all your constructor arguments in configs/vis. 
5. import your visualizer in the visualizers/__init__.py file.
6. Edit this README to include your new addition.
