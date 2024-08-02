import xml.etree.ElementTree as ET
import numpy as np

def parse_dae_file(dae_file_path):
    """
    Parse a COLLADA (.dae) file to extract vertex positions and calculate
    an axis-aligned bounding box.
    """
    tree = ET.parse(dae_file_path)
    root = tree.getroot()

    # COLLADA namespace
    ns = {'collada': 'http://www.collada.org/2005/11/COLLADASchema'}

    # Find the <float_array> element that contains vertex position data
    # This may need adjustment depending on the COLLADA file structure
    vertex_array = root.find(".//collada:source/collada:float_array", ns)

    if vertex_array is None:
        raise ValueError("Vertex position data not found in COLLADA file.")

    # Extract the vertex positions as a list of floats
    vertices = list(map(float, vertex_array.text.split()))

    # Reshape the list into an Nx3 NumPy array (N vertices, 3 coordinates each)
    vertices_np = np.array(vertices).reshape((-1, 3))

    # Calculate the axis-aligned bounding box
    min_coords = vertices_np.min(axis=0)
    max_coords = vertices_np.max(axis=0)

    return min_coords, max_coords

# Example usage
dae_file_path = '/home/andrei/.gazebo/models/person_standing/meshes/standing.dae'
min_coords, max_coords = parse_dae_file(dae_file_path)
print("Minimum coordinates:", min_coords)
print("Maximum coordinates:", max_coords)
