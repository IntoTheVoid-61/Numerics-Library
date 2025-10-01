from moment_of_inertia import *
import os
"""
The following example is for a cylinder with mass 1579.94g and
diameter (characteristic length) of 80mm.
"""


dir_path = os.path.dirname(os.path.realpath(__file__))
voxel_path = r"bodies\example_bodies\example_cylinder\example_cylinder.txt"
voxel_file = os.path.join(dir_path,voxel_path)

mass_body = 1579.94 * 10e-3
char_length_body = 80 * 10e-3

inertia_tensor = moment_of_inertia(voxel_file=voxel_file,
                                    mass=mass_body,
                                    char_length=char_length_body)

print(inertia_tensor)