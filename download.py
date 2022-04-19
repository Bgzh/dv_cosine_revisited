import os
import requests


def download(url: str, dest_folder: str, filename:str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    file_path = os.path.join(dest_folder, filename)
    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

if __name__ == "__main__":
    download("https://drive.google.com/file/d/1qxueBhd7WTBP58ZOdDL5K1DB0Sj2o5bZ/view?usp=sharing", "files_root", "alldata-id_p1gram.txt")
    download("https://drive.google.com/file/d/1GDttGJrnZh370Y0KNMbAMfRNU50La07R/view?usp=sharing", "files_root", "alldata-id_p3gram.txt")
    download("https://drive.google.com/file/d/1a-eOTfKXXqUpM19GepIxkZxI4N8ESSBJ/view?usp=sharing", "files_root", "train_vectors.txt")
    download("https://drive.google.com/file/d/1GFpVVrA1AlXBsWVx2McOnlAWyNm47TCI/view?usp=sharing", "files_root", "test_vectors.txt")
