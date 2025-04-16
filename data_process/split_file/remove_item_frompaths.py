def modify_dataloader_paths(input_file, output_file):
    """
    Replace 'dataset' with 'dataset_gaussian_airy' in all paths within the dataloader file
    
    Args:
        input_file (str): Path to the original dataloader file
        output_file (str): Path to save the modified dataloader file
    """
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Replace 'dataset' with 'dataset_gaussian_airy' 
    modified_content = content.replace('dataset', 'dataset_gaussian_airy')
    
    with open(output_file, 'w') as f:
        f.write(modified_content)
    
    print(f"Modified paths in {input_file} and saved to {output_file}")

if __name__ == "__main__":
    input_file = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataload_filename/eval_dataloader.txt"
    output_file = "/home/bingxing2/ailab/scxlab0061/Astro_SR/dataload_filename/eval_dataloader_gaussian_airy.txt"
    modify_dataloader_paths(input_file, output_file)