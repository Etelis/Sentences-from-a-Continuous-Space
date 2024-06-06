import os
import shutil
import pytest
from data.download_data import download_and_extract_data

@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    # Code to run before the test
    extract_to = "test_data"
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    yield
    # Code to run after the test
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)

def test_download_and_extract_data():
    url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
    extract_to = "test_data"
    
    # Run the download and extract function
    download_and_extract_data(url, extract_to)
    
    # Check if the files are created and not empty
    train_file = os.path.join(extract_to, 'data', 'ptb.train.txt')
    valid_file = os.path.join(extract_to, 'data', 'ptb.valid.txt')
    test_file = os.path.join(extract_to, 'data', 'ptb.test.txt')
    
    assert os.path.exists(train_file), f"{train_file} does not exist."
    assert os.path.getsize(train_file) > 0, f"{train_file} is empty."
    
    assert os.path.exists(valid_file), f"{valid_file} does not exist."
    assert os.path.getsize(valid_file) > 0, f"{valid_file} is empty."
    
    assert os.path.exists(test_file), f"{test_file} does not exist."
    assert os.path.getsize(test_file) > 0, f"{test_file} is empty."

if __name__ == "__main__":
    pytest.main([__file__])
