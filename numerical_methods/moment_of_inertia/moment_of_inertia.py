import numpy as np

"""


"""


def moment_of_inertia(voxel_file, char_length, mass):
    """
    Function calculates the inertia tensor at COM of the body.
    You can convert the .stl file of the body to voxel at: https://drububu.com/miscellaneous/voxelizer/?out=txt
    Export as .txt file.

    Parameters:
    ------------
    voxel_file: str
        The voxel.txt file of the body
    char_length: float
        The characteristic length of the body.
    mass: float
        Mass of the body.

    Returns:
    ---------
    inertia_tensor: numpy.ndarray
        The inertia tensor at COM.

    """
    coords = np.loadtxt(voxel_file, delimiter=',', dtype=int)
    coords = np.array([coord[:3] for coord in coords]) # Extract the first the elements in each row
    coords = coords/max(coords.ravel()) # Normalize
    x, y, z  = coords[:,0], coords[:,2], coords[:,1]

    x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z) # Geometric center of body, equal to COM if body is uniform
    x0, y0, z0 = x_mean, y_mean, z_mean # We will shift the body by negative of this vector, therefore COM is at (0,0,0)

    x,y,z = x - x0, y - y0, z - z0 # shift body

    num_of_voxels = len(x)

    Ixx = sum(y**2 + z**2)
    Iyy = sum(x**2 + z**2)
    Izz = sum(x**2 + y**2)
    Ixy = Iyx = -(sum(x*y))
    Ixz = Izx = -(sum(x*z))
    Iyz = Izy = -(sum(z*y))

    inertia_tensor = np.array([[Ixx,Ixy,Ixz],
                          [Iyx,Iyy,Iyz],
                          [Izx,Izy,Izz]]) / num_of_voxels
    
    inertia_tensor *= mass 
    inertia_tensor *= char_length**2 

    return inertia_tensor