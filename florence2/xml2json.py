from lxml import etree
import os 
import json 
import jsonlines
# 读取 xml 文件信息，并返回字典形式
def parse_xml_to_dict(xml):
    if len(xml) == 0:  # 遍历到底层，直接返回 tag对应的信息
        return {xml.tag: xml.text}
 
    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def convert(xml_path, img_path):


    # for index, xml_path in enumerate(os.listdir(xml_folder)):
    with open(xml_path, encoding='utf-8', errors='ignore') as fid:  # 防止出现非法字符报错
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = parse_xml_to_dict(xml)["annotation"]  # 读取 xml文件信息
    width = data['size']['width']
    height = data['size']['height']
    ob = []         # 存放目标信息
    res = {"query": "Find <ref-object>", 
        "response": "<bbox>", 
        "images":[img_path], #img_path
        "objects":[]
        }
    # object format: "[{\"caption\": \"guy in red\", \"bbox\": [138, 136, 235, 359], \"bbox_type\": \"real\", \"image\": 0}]"
    # image: bbox对应的图片是第几张, 索引从0开始
    for i in data['object']:        # 提取检测框
        name = str(i['name'])        # 检测的目标类别
        bbox = i['bndbox']
        xmin = int(bbox['xmin'])
        ymin = int(bbox['ymin'])
        xmax = int(bbox['xmax'])
        ymax = int(bbox['ymax'])
        # xmin = int(1000*(xmin/width))
        # ymin = int(1000*(ymin/height))
        # xmax = int(1000*(xmax/width))
        # ymax = int(1000*(ymax/height))
        # tmp = [name,xmin,ymin,xmax,ymax]    # 单个检测框
        tmp = {
        "caption": name,
        "bbox": [xmin, ymin, xmax, ymax],
        "bbox_type": "real",
        "image": 0
        }
        ob.append(tmp)
    res['objects'] = ob
    # template
    # {"query": "Find <ref-object>", "response": "<bbox>", "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [138, 136, 235, 359], \"bbox_type\": \"real\", \"image\": 0}]" }

    return res 


if __name__ == '__main__':
    # xml_path = '/home/oem/work/swift/asset/dataset/train/5uhyf.xml'
    # img_path = '/home/oem/work/swift/asset/dataset/train/5uhyf.jpg'
    xml_folder = '/home/oem/work/swift/asset/dataset/train/'
    save_path = 'data.jsonl'
    img_folder = xml_folder
    with jsonlines.open(save_path, 'w') as writer:

        
        for file_name in os.listdir(xml_folder):
            if not file_name.endswith('.xml') and not file_name.endswith('.db'):
                img_path = os.path.join(img_folder, file_name)
                name_without_ext = os.path.splitext(file_name)[0]
                # xml_path = os.path.join(xml_folder, file_name)
                # print(file_name)
                
                xml_path = os.path.join(xml_folder, name_without_ext+'.xml')

                res = convert(xml_path=xml_path, img_path=img_path)
                writer.write(res)
    # json_str = json.dumps(res)
    # print(json_str)
