import os
import requests
import tarfile
import shutil

def download_and_extract_data(url, extract_to):
    # Ensure the directory exists
    os.makedirs(extract_to, exist_ok=True)
    
    # Download the file
    response = requests.get(url)
    tar_path = os.path.join(extract_to, 'simple-examples.tgz')
    with open(tar_path, 'wb') as file:
        file.write(response.content)

    # Extract the tar file
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)

    # Move the relevant files to the data directory
    data_dir = os.path.join(extract_to, 'data')
    os.makedirs(data_dir, exist_ok=True)
    os.rename(os.path.join(extract_to, 'simple-examples/data/ptb.train.txt'), os.path.join(data_dir, 'ptb.train.txt'))
    os.rename(os.path.join(extract_to, 'simple-examples/data/ptb.valid.txt'), os.path.join(data_dir, 'ptb.valid.txt'))
    os.rename(os.path.join(extract_to, 'simple-examples/data/ptb.test.txt'), os.path.join(data_dir, 'ptb.test.txt'))

    # Clean up
    os.remove(tar_path)
    shutil.rmtree(os.path.join(extract_to, 'simple-examples'))

if __name__ == "__main__":
    url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
    extract_to = "."
    download_and_extract_data(url, extract_to)