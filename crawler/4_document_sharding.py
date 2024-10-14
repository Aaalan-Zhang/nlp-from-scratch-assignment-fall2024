import os

def shard_text(text, shard_size=500, overlap_percent=0.1):
    overlap_size = int(shard_size * overlap_percent)  # Calculate overlap size (10% of shard_size)
    shards = []

    # Create shards with overlap
    start = 0
    while start < len(text):
        end = start + shard_size
        shard = text[start:end]
        shards.append(shard)

        # Move start position forward with overlap
        start += (shard_size - overlap_size)
    
    return shards

def process_file(file_path, output_dir, shard_size=500, overlap_percent=0.1):
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Shard the text
    shards = shard_text(text, shard_size, overlap_percent)

    # Get the base name of the file without extension (e.g., "0" from "0.txt")
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Save each shard to a new file
    for i, shard in enumerate(shards):
        output_file = os.path.join(output_dir, f"{base_name}-{i}.txt")
        with open(output_file, 'w', encoding='utf-8') as shard_file:
            shard_file.write(shard)

    print(f"Processed {file_path}, created {len(shards)} shards.")

def shard_directory(input_dir, output_dir, shard_size=500, overlap_percent=0.1):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_dir, file_name)
            process_file(file_path, output_dir, shard_size, overlap_percent)

if __name__ == "__main__":
    input_directory = '/path/to/input/directory'  # Replace with the path to your input directory
    output_directory = '/path/to/output/directory'  # Replace with the path to your output directory

    # Start the sharding process
    shard_directory(input_directory, output_directory)