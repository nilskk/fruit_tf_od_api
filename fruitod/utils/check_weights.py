import glob
import json

if __name__ == '__main__':
    files_with_weights = 0
    files_without_weights = 0
    for weight_file in glob.glob('/data/classes/*/*.json'):
        with open(weight_file) as f:
            json_dict = json.load(f)
        if 'weightInGrams' in json_dict.keys():
            files_with_weights += 1
        else:
            files_without_weights += 1
            print(weight_file)

    print('Files with Weights: {}'.format(files_with_weights))
    print('Files without Weights: {}'.format(files_without_weights))
