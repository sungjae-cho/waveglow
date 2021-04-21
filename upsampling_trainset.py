from tqdm import tqdm
import os
import random

def upsampling(src_path, dst_path):
    with open(src_path, 'r') as f:
        paths = f.readlines()
    print('Last path:', paths[-1])

    spk_emo_paths = dict()
    print("Collect paths by (speaker, emotion) pairs...")
    for path in tqdm(paths):
        dir_path, wav_name = os.path.split(path)
        wav_name_splited = wav_name.split('_')
        if len(wav_name_splited) == 3:
            speaker = wav_name_splited[0]
            emotion = wav_name_splited[1]
        elif len(wav_name_splited) == 2:
            speaker = wav_name_splited[0]
            emotion = 'neutral'
        if (speaker, emotion) not in spk_emo_paths.keys():
            spk_emo_paths[(speaker, emotion)] = list()
        spk_emo_paths[(speaker, emotion)].append(path)

    len_list = list()
    for paths in spk_emo_paths.values():
        len_list.append(len(paths))
    max_paths = max(len_list)
    print("max_paths", max_paths)

    print("Each set of (speaker, emotion) is upsampled as follows.")
    all_new_paths = list()
    for spk_emo, paths in spk_emo_paths.items():
        new_paths = paths * (max_paths // len(paths))
        random.shuffle(paths)
        new_paths += paths[:(max_paths % len(paths))]
        all_new_paths += new_paths

        print("{}: {} >> {}".format(spk_emo, len(paths), len(new_paths)))

    all_new_paths = sorted(all_new_paths)
    str_new_paths = str()

    with open(dst_path, 'w') as f:
        f.writelines(all_new_paths)

    print("Upsampled {} is saved at {}".format(src_path, dst_path))

    with open(dst_path, 'r') as f:
        new_paths_2 = f.readlines()
    print("In the file, there are {} paths.".format(len(new_paths_2)))

if __name__ == "__main__":
    upsampling(
        'file_lists/02_KETTS_KETTS2_KSS_NC/train_files.txt',
        'file_lists/03_KETTS_KETTS2_KSS_NC_upsampled/train_files.txt'
    )
