import os
import h5py
import numpy as np

def load_and_average_encodings(root_folder, category, domains):
    """
    Load encodings for a category across all domains and compute the average encoding.

    Args:
        root_folder (str): Path to the root directory containing the domains.
        category (str): Category for which to load encodings.
        domains (list): List of domain folders.

    Returns:
        np.ndarray: Average encoding for the category across domains.
    """
    encodings_list = []

    for domain in domains:
        category_folder = os.path.join(root_folder, domain, category)
        parent_folder, current_folder = os.path.split(category_folder)
        parent_parent_folder, parent_folder_name = os.path.split(parent_folder)
        parent_folder_name_no_spaces = parent_folder_name.replace(' ', '_')
        current_folder_no_spaces = current_folder.replace(' ', '_')
        output_filename = f"{parent_folder_name_no_spaces}_{current_folder_no_spaces}.h5"

        
        if os.path.exists(category_folder):
            print("Cateogory path exists")
            file = os.path.join(category_folder, "SD_encodings", output_filename)
            if file.endswith(".h5") and category in file:
                with h5py.File(file, 'r') as hdf:
                    encodings = hdf['encodings'][:]
                    encodings_list.append(encodings)
    
    if encodings_list:
        combined_encodings = np.vstack(encodings_list)
        average_encoding = np.mean(combined_encodings, axis=0)
        return average_encoding
    else:
        print(f"No encodings found for category: {category}")
        return None

def assign_categories_to_clients(categories, num_clients):
    """
    Distribute categories among clients.

    Args:
        categories (list): List of category names.
        num_clients (int): Number of clients.

    Returns:
        dict: Mapping of client IDs to their assigned categories.
    """
    return {f"client_{i}": categories[i::num_clients] for i in range(num_clients)}

def process_clients(root_folder, domains, categories, num_clients):
    """
    Process encodings for each client by averaging encodings for their categories.

    Args:
        root_folder (str): Path to the root directory containing the domains.
        domains (list): List of domain folders.
        categories (list): List of all categories.
        num_clients (int): Number of clients.

    Returns:
        dict: Mapping of client IDs to their category average encodings.
    """
    client_assignments = assign_categories_to_clients(categories, num_clients)
    client_averages = {}

    for client, assigned_categories in client_assignments.items():
        client_averages[client] = {}
        for category in assigned_categories:
            average_encoding = load_and_average_encodings(root_folder, category, domains)
            if average_encoding is not None:
                client_averages[client][category] = average_encoding
            else:
                print(f"Skipping category {category} for {client} due to missing data.")
    
    return client_averages

def save_client_encodings(client_averages, output_folder):
    """
    Save each client's averaged encodings to an HDF5 file.

    Args:
        client_averages (dict): Mapping of client IDs to their category average encodings.
        output_folder (str): Path to the folder where output files will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for client, categories in client_averages.items():
        client_file = os.path.join(output_folder, f"{client}.h5")
        with h5py.File(client_file, 'w') as hdf:
            for category, encoding in categories.items():
                hdf.create_dataset(category, data=encoding)
        
        print(f"Averaged encodings saved for {client} in {client_file}")

if __name__ == "__main__":
    # Paths and parameters
    root_folder = "/proj/wasp-nest-cr01/users/x_obaza/oneshot_diff/NICO_DG"
    domains = ["outdoor", "autumn", "dim", "grass", "rock", "water"]  # Replace with actual domain names
    categories = os.listdir(os.path.join(root_folder, domains[0]))  # Replace with actual category names
    num_clients = 6
    output_folder = "/proj/wasp-nest-cr01/users/x_obaza/oneshot_diff/labelskew_encodings"  # Replace with desired output folder path

    # Process clients and save averaged encodings
    client_averages = process_clients(root_folder, domains, categories, num_clients)
    save_client_encodings(client_averages, output_folder)
