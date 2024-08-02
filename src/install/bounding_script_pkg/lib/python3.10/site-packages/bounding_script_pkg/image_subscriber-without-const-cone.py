import xml.etree.ElementTree as ET

def parse_sdf_file(sdf_file_path):
    # Parse the XML file
    tree = ET.parse(sdf_file_path)
    root = tree.getroot()

    # Initialize a dictionary to hold model information
    model_info = {}

    # Extract the model name
    model_name = root.find('.//model').get('name')
    model_info['model_name'] = model_name
    model_info['links'] = []

    # Iterate through each link in the model
    for link in root.findall('.//link'):
        link_name = link.get('name')
        link_info = {'link_name': link_name}

        # Iterate through each geometry in the link
        for geometry in link.findall('.//geometry/mesh'):
            uri = geometry.find('uri').text
            scale = geometry.find('scale').text if geometry.find('scale') is not None else "1 1 1"

            # Save the mesh URI and scale
            link_info['mesh_uri'] = uri
            link_info['scale'] = scale

        model_info['links'].append(link_info)

    return model_info

# Example usage
sdf_file_path = sdf_file_path = '/home/andrei/.gazebo/models/person_standing/model.sdf'  # Update this path
model_info = parse_sdf_file(sdf_file_path)
print(model_info)
