import pickle
import numpy as np

def explore_structure(data, indent=0):
    """
    Recursively explore and print the structure of a dictionary.
    
    Parameters:
    data (dict or list): The data to explore.
    indent (int): The current indentation level for printing.
    """
    # Set the indentation
    prefix = ' ' * indent

    if isinstance(data, dict):
        # Print all keys in the dictionary
        for key, value in data.items():
            print(f"{prefix}Key: {key}")
            explore_structure(value, indent + 4)
    elif isinstance(data, list):
        # Print information about the list
        print(f"{prefix}List of length {len(data)}")
        if len(data) > 0:
            if isinstance(data[0], (dict, list)):
                print(f"{prefix}First element structure:")
                explore_structure(data[0], indent + 4)
            else:
                # Print the type of the first element
                print(f"{prefix}First element type: {type(data[0])}")
    elif isinstance(data, np.ndarray):
        # Print the shape and type of the array
        print(f"{prefix}Array of shape {data.shape} and type {data.dtype}")
    else:
        # Print the type of the value
        print(f"{prefix}Type: {type(data)}")

def print_structure_from_pkl(file_path):
    # Open the pickle file in read binary mode
    with open(file_path, 'rb') as file:
        # Load the content of the file
        data = pickle.load(file)
        
        # Start exploring the structure
        explore_structure(data)

# Replace 'your_file.pkl' with the path to your .pkl file
print_structure_from_pkl('robosuite/scripts_okami/data/reach.pkl')
# print_structure_from_pkl('robosuite/scripts_okami/data/saved_states.pkl')
