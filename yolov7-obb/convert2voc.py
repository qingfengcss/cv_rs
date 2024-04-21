import os 
import shutil 
import xml.dom.minidom 
import xml.etree.ElementTree as ET 
from xml.dom.minidom import Document

def check_coord(x, h):
    x = 0 if x < 0 else x 
    x = h-1 if x >(h-1) else x 
    return x

def get_boxes(label_path, w, h):
    boxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            splitline = line.strip().split(' ')
            label = splitline[-1]
            x1 = check_coord(int(splitline[0]), h)
            y1 = check_coord(int(splitline[1]), w)
            x2 = check_coord(int(splitline[2]), h)
            y2 = check_coord(int(splitline[3]), w)
            x3 = check_coord(int(splitline[4]), h)
            y3 = check_coord(int(splitline[5]), w)
            x4 = check_coord(int(splitline[6]), h)
            y4 = check_coord(int(splitline[7]), w)
            box = [x1, y1, x2, y2, x3, y3, x4, y4, label, 0]
            boxes.append(box)
    return boxes




def write_xml(img_name, boxes, save_folder, w, h, d):
    # w = 512
    # h = 512
    # d = 3
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    annotation.appendChild(folder)

    folder_txt = doc.createTextNode('voc2007')
    folder.appendChild(folder_txt)
    
    file_name = doc.createElement('filename')
    annotation.appendChild(file_name)
    file_name_txt = doc.createTextNode(img_name)
    file_name.appendChild(file_name_txt)

    source = doc.createElement('source')
    annotation.appendChild(source)
    

    database = doc.createElement('database')
    source.appendChild(database)

    database_txt = doc.createTextNode("My dataset")
    database.appendChild(database_txt)

    annotation_new = doc.createElement('annotation')
    source.appendChild(annotation_new)
    annotation_new_txt = doc.createTextNode('voc2007')
    annotation_new.appendChild(annotation_new_txt)

    image = doc.createElement('image')
    source.appendChild(image)
    image_txt = doc.createTextNode("flickr")
    image.appendChild(image_txt)

    
    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid = doc.createElement('flickrid')
    owner.appendChild(flickrid)
    flickrid_txt = doc.createTextNode("NULL")
    flickrid.appendChild(flickrid_txt)
    
    ow_name = doc.createElement('name')
    owner.appendChild(ow_name)
    ow_name_txt = doc.createTextNode("idannel")
    ow_name.appendChild(ow_name_txt)

    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(w))
    width.appendChild(width_txt)

    height = doc.createElement('height')
    size.appendChild(height)
    height_txt = doc.createTextNode(str(h))
    height.appendChild(height_txt)

    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode(str(d))
    depth.appendChild(depth_txt)
            
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented_txt = doc.createTextNode("1")
    segmented.appendChild(segmented_txt)

    for bbox in boxes:
        # threes#
        object_new = doc.createElement("object")
        annotation.appendChild(object_new)

        name = doc.createElement('name')
        object_new.appendChild(name)
        name_txt = doc.createTextNode(str(bbox[-2]))
        name.appendChild(name_txt)

        pose = doc.createElement('pose')
        object_new.appendChild(pose)
        pose_txt = doc.createTextNode("Unspecified")
        pose.appendChild(pose_txt)

        truncated = doc.createElement('truncated')
        object_new.appendChild(truncated)
        truncated_txt = doc.createTextNode("0")
        truncated.appendChild(truncated_txt)

        difficult = doc.createElement('difficult')
        object_new.appendChild(difficult)
        difficult_txt = doc.createTextNode(str(bbox[-1]))
        difficult.appendChild(difficult_txt)
        # threes-1#
        bndbox = doc.createElement('rotated_bndbox')
        object_new.appendChild(bndbox)

        x0 = doc.createElement('x1')
        bndbox.appendChild(x0)
        x0_txt = doc.createTextNode(str(bbox[0]))
        x0.appendChild(x0_txt)
        y0 = doc.createElement('y1')
        bndbox.appendChild(y0)
        y0_txt = doc.createTextNode(str(bbox[1]))
        y0.appendChild(y0_txt)
        x1 = doc.createElement('x2')
        bndbox.appendChild(x1)
        x1_txt = doc.createTextNode(str(bbox[2]))
        x1.appendChild(x1_txt)
        y1 = doc.createElement('y2')
        bndbox.appendChild(y1)
        y1_txt = doc.createTextNode(str(bbox[3]))
        y1.appendChild(y1_txt)
        x2 = doc.createElement('x3')
        bndbox.appendChild(x2)
        x2_txt = doc.createTextNode(str(bbox[4]))
        x2.appendChild(x2_txt)
        y2 = doc.createElement('y3')
        bndbox.appendChild(y2)
        y2_txt = doc.createTextNode(str(bbox[5]))
        y2.appendChild(y2_txt)
        x3 = doc.createElement('x4')
        bndbox.appendChild(x3)
        x3_txt = doc.createTextNode(str(bbox[6]))
        x3.appendChild(x3_txt)
        y3 = doc.createElement('y4')
        bndbox.appendChild(y3)
        y3_txt = doc.createTextNode(str(bbox[7]))
        y3.appendChild(y3_txt)
    
    xmlname = os.path.splitext(img_name)[0]
    tempfile = os.path.join(save_folder, xmlname + '.xml')
    print(tempfile)
    with open(tempfile, 'wb') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))


def parse_folder(img_folder, label_folder, save_folder, w, h, d):
    
    for img_name in os.listdir(img_folder):
        label_path = os.path.join(label_folder, img_name.split('.')[0]+'.txt')
        boxes = get_boxes(label_path, w, h)
        write_xml(img_name, boxes, save_folder, w, h, d)



if __name__ == '__main__':
    # img_name = '0000.png'
    # save_folder = '/home/chengshunsheng/competition/datasets/Annotations'
    # label_path = '/home/chengshunsheng/competition/labels/train/0000.txt'
    # w = 512
    # h = 512
    # d = 3
    # boxes = get_boxes(label_path)
    # write_xml(img_name, boxes, save_folder, w, h, d)
    img_folder = '/home/chengshunsheng/competition/data/images/val'
    label_folder = '/home/chengshunsheng/competition/labels/val'
    save_folder = '/home/chengshunsheng/competition/datasets/VOC2007/Annotations'
    w = 512
    h = 512
    d = 3
    parse_folder(img_folder, label_folder, save_folder, w, h, d)