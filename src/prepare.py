import yaml
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

params = yaml.safe_load(open('params.yaml'))['prepare']

split = params['split']
seed = params['seed']
shuffle = params['shuffle']
stratify = params['stratify']

X = pd.read_csv(os.path.join('data', 'raw', '_annotations.csv'))
y = X['label']
if stratify:
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split, random_state=seed, shuffle=shuffle, stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split, random_state=seed, shuffle=shuffle)

output_train = os.path.join('data', 'prepared', 'train')
output_test = os.path.join('data', 'prepared', 'test')

for label in X['label'].unique():
    if not os.path.exists(f'data/processed/train/{label}'):
        os.makedirs(f'data/processed/train/{label}')
    if not os.path.exists(f'data/processed/test/{label}'):
        os.makedirs(f'data/processed/test/{label}')

for i in X_train.index:
    image = Image.open(f'data/raw/{X_train.filename[i]}')
    image = image.crop((X_train.xmin[i],X_train.ymin[i],X_train.xmax[i],X_train.ymax[i]))
    X_train.at[i,'width'], X_train.at[i,'height'] = image.size
    image.save(f'data/processed/train/{X_train.label[i]}/{X_train.filename[i]}')
X_train.drop(columns=['xmin', 'ymin', 'xmax', 'ymax'], inplace=True)
X_train.to_csv('data/processed/train/_annotations.csv', index=False)

for i in X_test.index:
    image = Image.open(f'data/raw/{X_test.filename[i]}')
    image = image.crop((X_test.xmin[i],X_test.ymin[i],X_test.xmax[i],X_test.ymax[i]))
    X_test.at[i,'width'], X_test.at[i,'height'] = image.size
    image.save(f'data/processed/test/{X_test.label[i]}/{X_test.filename[i]}')
X_test.drop(columns=['xmin', 'ymin', 'xmax', 'ymax'], inplace=True)
X_test.to_csv('data/processed/test/_annotations.csv', index=False)
