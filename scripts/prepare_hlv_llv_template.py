import h5py

# File names
file_hlv = '/home/users/o/oleksiyu/WORK/hyperproject/workspaces/2024_09_09/TRANSITv1_HLV/template/outputs/template_sample_con.h5'
file_llv = '/home/users/o/oleksiyu/WORK/hyperproject/workspaces/2024_09_09/TRANSIT_LLV/template/outputs/template_sample_con.h5'
file_c = '/home/users/o/oleksiyu/WORK/hyperproject/user/llv_hlv_templates/a.h5'

# Open the new file (c.h5) for writing
with h5py.File(file_c, 'w') as c_file:
    # Open and copy datasets from a.h5
    with h5py.File(file_a, 'r') as a_file:
        for dataset_name in a_file:
            a_file.copy(a_file[dataset_name], c_file, name=dataset_name)
    
    # Open and copy datasets from b.h5
    with h5py.File(file_b, 'r') as b_file:
        for dataset_name in b_file:
            # Ensure unique names by adding a prefix or check for name clashes
            if dataset_name in c_file:
                new_dataset_name = f"{dataset_name}_from_b"
                b_file.copy(b_file[dataset_name], c_file, name=new_dataset_name)
            else:
                b_file.copy(b_file[dataset_name], c_file, name=dataset_name)

print(f"Merged datasets from {file_a} and {file_b} into {file_c}.")