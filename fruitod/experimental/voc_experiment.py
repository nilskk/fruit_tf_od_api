import os
from pathlib import Path
import shutil
import random
import glob

def list_difference(list1, list2):
    difference = [value for value in list1 if value not in list2]
    return difference

def write_lists(list, txt_path):
    with open(txt_path, 'w') as f:
        f.writelines('\n'.join(list))

def make_voc_structure(input_path, output_path, train_list, val_list):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    trainval_path = os.path.join(output_path, 'ImageSets/Main')
    Path(trainval_path).mkdir(parents=True, exist_ok=True)
    image_path = os.path.join(output_path, 'JPEGImages')
    Path(image_path).mkdir(parents=True, exist_ok=True)
    annotation_path = os.path.join(output_path, 'Annotations')
    Path(annotation_path).mkdir(parents=True, exist_ok=True)

    write_lists(train_list, os.path.join(trainval_path, 'vott_train.txt'))
    write_lists(val_list, os.path.join(trainval_path, 'vott_val.txt'))
    shutil.copytree(os.path.join(input_path, 'JPEGImages'), image_path, dirs_exist_ok=True)
    copy_annotations(os.path.join(input_path, 'Annotations'), annotation_path, train_list)
    copy_annotations(os.path.join(input_path, 'Annotations'), annotation_path, val_list)
    shutil.copy2(os.path.join(input_path, 'pascal_label_map.pbtxt'), output_path)

def copy_annotations(input_path, output_path, image_list):
    annotation_list = [Path(value).with_suffix('.xml') for value in image_list]
    for annotation_file in annotation_list:
        shutil.copy2(os.path.join(input_path, annotation_file), output_path)

def get_class_list():
    apfel_list = []
    kiwi_list = []
    paprikamix_list = []
    paprikarot_list = []
    kohlrabi_list = []
    mango_list = []
    tomate_list = []
    for file in glob.glob('/data/voc_data_penny/classes/*/images/*'):
        if 'apfel' in file:
            apfel_list.append(Path(file).name)
        if 'bunte_paprika' in file:
            paprikamix_list.append(Path(file).name)
        if 'kiwi' in file:
            kiwi_list.append(Path(file).name)
        if 'kohlrabi' in file:
            kohlrabi_list.append(Path(file).name)
        if 'mango' in file:
            mango_list.append(Path(file).name)
        if 'paprika_rot' in file:
            paprikarot_list.append(Path(file).name)
        if 'tomate' in file:
            tomate_list.append(Path(file).name)

    random.shuffle(apfel_list)
    random.shuffle(kiwi_list)
    random.shuffle(paprikamix_list)
    random.shuffle(paprikarot_list)
    random.shuffle(mango_list)
    random.shuffle(tomate_list)
    random.shuffle(kohlrabi_list)

    train1_list = []
    val1_list = []
    val2_list = []
    val3_list = []
    for list in [apfel_list, kiwi_list, paprikamix_list, paprikarot_list, mango_list, tomate_list, kohlrabi_list]:

        tmp_list = random.sample(list, 200)
        train1_list = train1_list + tmp_list
        remaining_list = list_difference(list, tmp_list)

        tmp_list = random.sample(remaining_list, 50)
        val1_list = val1_list + tmp_list
        remaining_list = list_difference(list, tmp_list)

        tmp_list = random.sample(remaining_list, 50)
        val2_list = val2_list + tmp_list
        remaining_list = list_difference(list, tmp_list)

        val3_list = val3_list + remaining_list

        return train1_list, val1_list, val2_list, val3_list




if __name__ == '__main__':
    voc_input_path = Path('/data/voc_data_penny')
    experiment_output = Path('/data/experiment')
    experiment_output.mkdir(parents=True, exist_ok=True)

    image_list = os.listdir(os.path.join(voc_input_path, 'JPEGImages'))
    
    train1_list, val1_list, val2_list, val3_list = get_class_list()

    train2_list = train1_list + val1_list
    train3_list = train2_list + val2_list

    # Gesamtdatensatz
    main_path = os.path.join(experiment_output, 'voc_gesamt')
    make_voc_structure(voc_input_path, main_path, train3_list, val3_list)


    # Experiment 1
    main_path = os.path.join(experiment_output, 'voc_experiment_1')
    make_voc_structure(voc_input_path, main_path, train1_list, val1_list)

    # Experiment 2
    main_path = os.path.join(experiment_output, 'voc_experiment_2')
    make_voc_structure(voc_input_path, main_path, train1_list, val2_list)

    # Experiment 3
    main_path = os.path.join(experiment_output, 'voc_experiment_3')
    make_voc_structure(voc_input_path, main_path, train1_list, val3_list)

    

            