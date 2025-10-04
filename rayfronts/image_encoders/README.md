# Image Encoders

This directory includes all different 2D image encoders consuming RGB images and
producing semantic features of varios kinds.
All encoders follow a template from the base.py file which provides documented abstract classes.

## Available Options:
- NARADIO: The RayFronts encoder and is the best performing encoder for zero-shot 3D semantic segmentation as shown in the paper.
- RADIO: https://github.com/NVlabs/RADIO/tree/main
- NACLIP: https://github.com/sinahmr/NACLIP
- Trident: https://github.com/YuHengsss/Trident.git (Requires extra dependencies)
- ConceptFusion: https://github.com/concept-fusion/concept-fusion (Requires extra dependencies).

## Adding an Image Encoder
0. Read the [CONTRIBUTING](../../CONTRIBUTING.md) file.
1. Create a new python file with the same name as your encoder.
2. Extend one of the base abstract classes found in [base.py](base.py).
3. Implement and override the inherited methods.
4. Add a config file with all your constructor arguments in configs/encoder. 
5. import your encoder in the image_encoders/__init__.py file.
6. Edit this README to include your new addition.
7. If your encoder relies on multiple supporting files, then 

    - Option 1: Add it as a submodule and only have a single stub file in this
  repo which contains the encoder class defined by one of our base abstract classes. 
    - Option 2: If very few supporting files are needed, add it as a directory which has the same name X as the encoder. Inside the directory there should be an X.py which defines your encoder class and an __init__.py which imports your encoder class such that its available in the X module not the X.X module.
