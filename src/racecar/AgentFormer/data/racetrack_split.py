import os

def get_racetrack_split(dataset, dataset_mode):
    dir = f'datasets/{dataset}/{dataset_mode}'
    files = os.listdir(dir)

    train, val, test = [], [], []
    for file in files:
        basename = os.path.splitext(file)[0]
        if 'labels' in basename or 'mapping' in basename or 'ego_ids' in basename:
            continue
        if 'train' in basename:
            train.append(basename)
        elif 'val' in basename:
            val.append(basename)
        else:
            test.append(basename)

    return train, val, test