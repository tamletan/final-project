import os
import re
import pandas as pd

# part = 'enron6'
# root = r'..\dataset\data-train\enron\{}'.format(part)
# csv_file = r'.\{}.csv'.format(part)

root = r'..\dataset\data-train\corpus'
csv_file = r'.\corpus.csv'


def get_file(dataset_path):
    f = []
    for dirs, _, files in os.walk(dataset_path):
        if files:
            for file in files:
                f.append(os.path.join(dirs, file))

    sp = []
    lg = []

    for path in f:
        tag = path.split(os.path.sep)[-2]
        if tag == 'ham':
            lg.append(path)
        if tag == 'spam':
            sp.append(path)

    return sp, lg


def read_file(dataset_path):
    spam, legit = get_file(dataset_path)

    tags = []
    bodys = []

    for file in spam:
        with open(file, 'r') as f:
            try:
                data = f.read()
                body = re.sub('Subject:.*\n', '', data)
                bodys.append(body)
                tags.append('spam')
            except:
                print(file)

    for file in legit:
        with open(file, 'r') as f:
            try:
                data = f.read()
                body = re.sub('Subject:.*\n', '', data)
                bodys.append(body)
                tags.append('legit')
            except:
                print(file)

    data = {'tag': tags, 'body': bodys}
    return data


if __name__ == '__main__':
    data = read_file(root)

    df = pd.DataFrame(data, columns = ['tag', 'body'])
    print(df['tag'].value_counts())
    df.to_csv(csv_file, index=False)
