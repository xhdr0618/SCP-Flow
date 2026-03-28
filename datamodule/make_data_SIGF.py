"""
convert SIGF database into compressed npz dataset with img/time/label seqs(SIGF_make)

this implementation is based on https://github.com/ZhangYH0502/C2F-LDM/blob/master/make_data.py

Example of usage:
python make_data_SIGF.py --input_database_path /path/to/SIGF-database --output_make_path /path/to/output/SIGF_make

"""

import numpy as np
from PIL import Image
import os
import datetime
import argparse

"""
pack each patient into a npz file with keys: seq_imgs, times, labels
"""
def sort_by_date(image_list):
    """
    Sort the image list based on dates in filenames
    Date format: YYYY_MM_DD (e.g., 2004_08_04 in SD2204_2004_08_04_OS.JPG)
    """

    def extract_date(filename):
        # Split filename to get date parts
        parts = filename.split('_')
        if len(parts) >= 4:  # Ensure filename format is correct
            try:
                year = int(parts[1])
                month = int(parts[2])
                day = int(parts[3].split('_')[0])  # Handle potential suffix after day

                # Return date tuple for sorting
                return (year, month, day)
            except (ValueError, IndexError):
                # Return default minimum value if parsing fails
                return (0, 0, 0)
        return (0, 0, 0)  # Default minimum value

    # Sort using extracted date as key
    sorted_images = sorted(image_list, key=extract_date)
    return sorted_images

def read_text(file_name):
    x = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            x.append(line)
    return x


def write_text(file_name, file_content_1=None):
    log_file = open(file_name, "a")
    for c1 in file_content_1:
        log_file.write(c1 + '\n')
    log_file.close()


def compute_time_interval(t1, t2, time_method='year'):
    """
    Calculate time interval between two dates, rounded to year or month
    """
    d1 = datetime.date(int(t1[:4]), int(t1[4:6]), int(t1[6:]))
    d2 = datetime.date(int(t2[:4]), int(t2[4:6]), int(t2[6:]))
    days = (d2 - d1).days

    if time_method == 'year':
        return round(days / 365)
    elif time_method == 'month':
        return round(days / 30)
    else:
        raise ValueError("time_method must be 'year' or 'month'")


def SIGF_single(mode='train', input_database_path=None, output_make_path=None):
    """ this method can be used for constrcu VQGAN pretrain dataset and image pretrain dataset """
    img_path = os.path.join(input_database_path, mode, 'image')
    lab_path = os.path.join(input_database_path, mode, 'label')
    
    out_path = os.path.join(output_make_path, mode)
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    patients = os.listdir(img_path)

    for patient in patients:

        patient_path = os.path.join(img_path, patient)

        seq_images = os.listdir(patient_path)
        seq_len = len(seq_images)

        # seq_labels = read_text(os.path.join(lab_path, patient + '.txt'))

        for idx in range(seq_len):

            seq_name = seq_images[idx]

            fundus = Image.open(os.path.join(patient_path, seq_name))

            # label = int(seq_labels[idx])

            fundus.save(os.path.join(out_path, seq_name))


def SIGF(mode='train', input_database_path=None, output_make_path=None):

    max_time = 0
    pos_num = 0
    neg_num = 0

    img_path = os.path.join(input_database_path, mode, 'image')
    lab_path = os.path.join(input_database_path, mode, 'label')

    out_path = os.path.join(output_make_path, mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    patients = os.listdir(img_path)

    for patient in patients:

        patient_path = os.path.join(img_path, patient)

        seq_images = os.listdir(patient_path)
        seq_len = len(seq_images)
        seq_images = sort_by_date(seq_images)

        seq_labels = read_text(os.path.join(lab_path, patient+'.txt'))

        image_list = []
        time_list = []
        label_list = []
        """ For each patient, select a window of length 6, to get image sequences of length 6 """
        for idx in range(seq_len):

            seq_name = seq_images[idx]

            fundus = Image.open(os.path.join(patient_path, seq_name))
            fundus = fundus.resize((256, 256), Image.LANCZOS)
            fundus = np.array(fundus, dtype=np.float32)
            image_list.append(fundus)

            seq_name_split = seq_name.split('_')
            seq_time = seq_name_split[1] + seq_name_split[2] + seq_name_split[3]
            time_list.append(seq_time)

            label_list.append(int(seq_labels[idx]))


        max_num = len(image_list) - 6 + 1
        for i in range(max_num):

            image_s = image_list[i:i+6]
            time_s = time_list[i:i+6]
            label_s = label_list[i:i+6]
            
            if label_s[-1] == 1:
                pos_num = pos_num + 1
            else:
                neg_num = neg_num + 1

            time_s_new = [int(0)]
            for j in range(5):
                time_tmp = compute_time_interval(time_s[0], time_s[j+1],time_method="year")  # Get time interval (between first year and jth year), in years
                time_s_new.append(time_tmp)
            if time_s_new[5] > max_time:
                max_time = time_s_new[5]

            image_s = np.array(image_s)
            label_s = np.array(label_s)
            time_s_new = np.array(time_s_new)

            out_name = patient + '_' + str(i)

            np.savez(
                os.path.join(out_path, out_name + '.npz'),
                seq_imgs=image_s,
                times=time_s_new,
                labels=label_s,
            )
    
    print(f"Positive samples: {pos_num}")
    print(f"Negative samples: {neg_num}")


def parse_args():
    parser = argparse.ArgumentParser(description='Process SIGF fundus image data')
    
    parser.add_argument('--input_database_path', type=str, required=True,
                        help='Path to the input SIGF database, e.g., data/SIGF-database/')
    
    parser.add_argument('--output_make_path', type=str, required=True,
                        help='Path to save processed output, e.g., data/SIGF_make/')
    
    parser.add_argument('--modes', nargs='+', default=['train', 'validation', 'test'],
                        help='Processing modes (default: train validation test)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    for mode in args.modes:
        print(f"Processing {mode} data...")
        SIGF(mode=mode, 
             input_database_path=args.input_database_path, 
             output_make_path=args.output_make_path)