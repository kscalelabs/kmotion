# kmotion

Welcome to the kmotion project!

# Installation

Install blender instructions: https://www.blender.org/download/

Optional: Install blender into `blender_install_here` folder so that the paths for commands are the same


Make a conda env 
```
conda create -n kmotion python=3.11
```

Install the package 
```
pip install -e .
```



# Usage 


To plot the x y z values of the hip joint in global coordinates from the example bvh file, run 

```
python kmotion/scripts/read_bvh.py
```

Run default humanoid (this will apply the motion from the bvh file)
```
python kmotion/scripts/run_default_humanoid.py
# or 
python kmotion/scripts/run_default_humanoid.py --bvh-path /path/to/file.bvh
```

## Convert fbx to bvh 

```
$ ./path/to//blender -b -P kmotion/scripts/fbx2bvh.py -- --fbx_dir /path/to/fbx_dir
```
So to convert the files in the example script, run

```
$ ./blender_install_here/blender-4.4.0-linux-x64/blender -b -P kmotion/scripts/fbx2bvh.py -- --fbx_dir tests/assets/walk-relaxed_287304
```


# Details 

Blender FBX to BVH conversion script original source and issue: https://github.com/DeepMotionEditing/deep-motion-editing/issues/25#issuecomment-639322599



