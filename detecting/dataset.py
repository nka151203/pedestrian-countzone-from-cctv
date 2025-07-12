import os
import shutil
import configparser
import pandas as pd

#You can edit this folder dataset if you want
train_dir = ".\\detecting\\MOT20\\train"
test_dir = ".\\detecting\\MOT20\\test"

if os.path.exists(train_dir) and os.path.exists(test_dir):
    print("Datatset has downloaded")
else:
    print("Datatset hasn't downloaded. Please download and try again")
    exit()

def convert_to_yolo_format(bb, img_w, img_h):
  """
  Yolo format use center point, width and height to display bounding box
  MOT17 use top left corner, so you need to convert to yolo format.
  
  bb = (bb_left, bb_top, bb_width, bb_height): bounding box
  img_w: image width
  img_h: image height
  """
  #Original Formatt: (x_top_left, y_top_left, bb_w, bb_h)
  #Converted Format for YOLO: (id = 0, x_center, y_center, bb_w, bb_h)
  #id = 0 because we only detect pedestrian
  
  x_center = bb['bb_left'] + (bb['bb_width']/2)
  y_center = bb['bb_top'] + (bb['bb_height']/2)

  #Normalization to (0:1) scale
  x_center = x_center/img_w
  y_center = y_center/img_h
  bb_w = bb['bb_width']/img_w
  bb_h = bb['bb_height']/img_h

  return (0, x_center, y_center, bb_w, bb_h)

def process_a_folder(folder_path):
    """
    folder_path: a part folder of train/test. Ex: MOT20-01
    One frame detection information will have a detection file [frame_number].txt.
    """
    
    #Read image information from seqinfo.ini
    config = configparser.ConfigParser()
    print(folder_path)
    config.read(os.path.join(folder_path, 'seqinfo.ini'))
    img_w = int(config['Sequence']['imWidth'])
    img_h = int(config['Sequence']['imHeight'])
    
    #Read detection information
    det_path = os.path.join(folder_path, 'det/det.txt')
    det_data = pd.read_csv(
        det_path,
        header=None,
        names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class', 'visibility', 'dummy']
    )
    
    # Create labels folders for each frame
    labels_folder = os.path.join(folder_path, 'labels')
    os.makedirs(labels_folder, exist_ok=True)

    #Extract dt.txt to each frame detection. Ex: 000001.txt is label for first frame.
    for frame_number in det_data['frame'].unique():
        frame_data = det_data[det_data['frame'] == frame_number]
        label_file = os.path.join(labels_folder, f'{int(frame_number):06d}.txt')
        with open(label_file, 'w') as file:
            for _, row in frame_data.iterrows():
                yolo_bb = convert_to_yolo_format(row, img_w, img_h)
                file.write(f'{yolo_bb[0]} {yolo_bb[1]:.6f} {yolo_bb[2]:.6f} {yolo_bb[3]:.6f} {yolo_bb[4]:.6f}\n')

def process_list_folder(folder):
    """
    Apply process_a_folder for each part folder in train/test
    """
    for seq_folder in os.listdir(folder):
        seq_path = os.path.join(folder, seq_folder)
        if os.path.isdir(seq_path):
            process_a_folder(seq_path) 
    print("Processed: ",folder)

process_list_folder(train_dir)
process_list_folder(test_dir)


def rename_and_move(prefix_name, folder_get, dest_folder, endwith):
    if os.path.exists(folder_get) == False or "/images/" in folder_get or "/labels/" in folder_get:
        return 0
    else:
        for filename in os.listdir(folder_get):
            if filename.lower().endswith(endwith):
                old_path = os.path.join(folder_get, filename)
                new_filename = prefix_name + "-" + filename
                new_path = os.path.join(folder_get, new_filename)
                os.rename(old_path, new_path)
                shutil.move(new_path, dest_folder)
        return 1
    
def setup_dataset(based_folder):
    img_data = os.path.join(based_folder,"images")
    labels_data = os.path.join(based_folder,"labels")
    os.makedirs(img_data, exist_ok=True)
    os.makedirs(labels_data, exist_ok=True)
    for seq_folder in os.listdir(based_folder):
        seq_path = os.path.join(based_folder, seq_folder)
        get_img = os.path.join(seq_path, "img1")
        get_labels = os.path.join(seq_path, "labels")
        rename_and_move(seq_folder,get_img,img_data,".jpg")
        if rename_and_move(seq_folder,get_labels,labels_data,".txt") == 1:
            shutil.rmtree(seq_path)
    print("Data has done")

print("Preparing dataset for YOLO training")    
setup_dataset(train_dir)
setup_dataset(test_dir)
print("Dataset for Yolo training has been prepared")

