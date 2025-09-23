import os

test_file = "data.html"


def chunks(file_name, size=1000000):
    with open(file_name) as f:
        while content := f.read(size):
            yield content


relative_directory_path = "./chunked_files"

os.makedirs(relative_directory_path, exist_ok=True)


if __name__ == '__main__':
    split_files = chunks(test_file)
    for chunk in split_files:
        # create a new file for each chunk
        with open(f"chunk_{os.urandom(4).hex()}.txt", "w") as f:
            f.write(chunk)
        print(len(chunk))
