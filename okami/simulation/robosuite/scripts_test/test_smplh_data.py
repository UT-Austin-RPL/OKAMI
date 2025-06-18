import pickle

file_path = 'robosuite/scripts_okami/data/reach.pkl'

with open(file_path, 'rb') as file:
    # Load the content of the file
    data = pickle.load(file)

d = data[0]

print('left fingers =', d['left_fingers'][:, :3, 3])

print('right fingers =', d['right_fingers'][:, :3, 3])