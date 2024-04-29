
from pathlib import Path
import zipfile
import requests

# 1. Get data

my_data_path = Path('my_data')

images_path = my_data_path / 'my_pizza_steak_sushi'
zip_name = 'my_pizza_steak_sushi.zip'

if images_path.is_dir():
    print(f'{images_path} exists')
else:
    print(f'Can\'t find {images_path}, I will create one ...')
    images_path.mkdir(parents=True, exist_ok=True)

with open(my_data_path / zip_name, 'wb') as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print('downloading files ...')
    f.write(request.content)

with zipfile.ZipFile(my_data_path / zip_name) as f:
    print('Unzippping data ...')
    f.extractall(images_path)                                                                                                                                                                                                                                                                                                                                                                                                                                            
