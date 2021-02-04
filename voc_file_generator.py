import os
from pathlib import Path
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format


def convert_classes(classes, start=1):
    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=start):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    return text


if __name__ == '__main__':
    jpeg_path = Path('./data/voc_data/JPEGImages')
    output_trainval = Path('./test/ImageSets/Main/')
    output_trainval.mkdir(parents=True, exist_ok=True)
    output_labelmap = Path('./test/')
    output_labelmap.mkdir(parents=True, exist_ok=True)

    val_list=[]
    train_list=[]
    for i, image in enumerate(os.listdir(jpeg_path)):
        if i % 5 == 0:
            val_list.append(image + '\n')
        else:
            train_list.append(image + '\n')

    print(val_list)

    with open(os.path.join(output_trainval, 'vott_train.txt'), 'w') as f:
        f.writelines(train_list)

    with open(os.path.join(output_trainval, 'vott_val.txt'), 'w') as f:
        f.writelines(val_list)

    txt = convert_classes(['apfel', 'paprika_mix', 'kiwi', 'Kohlrabi', 'mango', 'rote paprika', 'tomate'])
    print(txt)
    with open(os.path.join(output_labelmap, 'pascal_label_map.pbtxt'), 'w') as f:
        f.write(txt)
