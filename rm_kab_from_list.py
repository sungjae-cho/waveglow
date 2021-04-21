from tqdm import tqdm

def rm_DB(fl_path, save_path, db2rm):
    str_new_paths = str()
    with open(fl_path, 'r') as f:
        paths = f.readlines()
    prefix = '/data2/sungjaecho/data_tts/'
    i = len(prefix)

    new_paths = list()
    print("Excluding {} paths...".format(db2rm))
    for path in tqdm(paths):
        if db2rm != path[i:i+len(db2rm)]:
            new_paths.append(path)

    print("len(paths)", len(paths))
    print("len(new_paths)", len(new_paths))

    with open(save_path, 'w') as f:
        f.writelines(new_paths)

    print("The file list without {} is saved at {}".format(db2rm, save_path))

if __name__ == "__main__":
    rm_DB(
        'file_lists/01_KETTS_KETTS2_KSS_NC_KAB/train_files.txt',
        'file_lists/02_KETTS_KETTS2_KSS_NC/train_files.txt',
        'KAB')
    rm_DB(
        'file_lists/01_KETTS_KETTS2_KSS_NC_KAB/test_files.txt',
        'file_lists/02_KETTS_KETTS2_KSS_NC/test_files.txt',
        'KAB')
