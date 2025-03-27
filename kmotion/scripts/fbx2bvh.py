"""Tool for converting FBX animation files to BVH format using Blender."""

import argparse
import os
import sys

import bpy
import numpy as np


def fbx2bvh(file_path: str) -> None:
    """Import an FBX file from file_path, export its animation as a BVH file.

    in the same directory, and then remove the imported action.

    Args:
        file_path: Path to the input FBX file
    """
    dir_path, filename = os.path.split(file_path)
    base, ext = os.path.splitext(filename)
    bvh_path = os.path.join(dir_path, base + ".bvh")

    # Import the FBX file
    bpy.ops.import_scene.fbx(filepath=file_path)

    # Get the imported action and determine its frame range
    action = bpy.data.actions[-1]
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])
    frame_end = int(np.max([60, frame_end]))  # Ensure at least 60 frames

    # Export the animation as a BVH file
    bpy.ops.export_anim.bvh(filepath=bvh_path, frame_start=frame_start, frame_end=frame_end, root_transform_only=True)

    # Clean up by removing the action
    bpy.data.actions.remove(bpy.data.actions[-1])
    print(f"Processed: {file_path} --> {bvh_path}")


def process_directory(root_dir: str) -> None:
    """Recursively search for all .fbx files under root_dir and process each one."""
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".fbx"):
                file_path = os.path.join(root, file)
                fbx2bvh(file_path)


def main() -> None:
    # Parse command-line arguments after the double dash "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Recursively convert FBX files to BVH files in Blender.")
    parser.add_argument("--fbx_dir", type=str, default="./fbx/", help="Directory to search for FBX files.")
    args = parser.parse_args(argv)

    data_path = args.fbx_dir
    print(f"Processing FBX files under directory: {data_path}")
    process_directory(data_path)


if __name__ == "__main__":
    main()
