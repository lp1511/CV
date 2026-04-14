import os
import pandas as pd
import re

image_extensions = {'.jpg', '.jpeg'}
folder_path = './datasets/data/data_all'
folder_path ='./datasets/data/data_all/output'
file_name = []
label = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        ext = os.path.splitext(filename)[1].lower()

        if ext in image_extensions:
            lnum = filename.replace(ext, '')
            lnum = lnum.split('_')[0]
            lnum = re.sub(r'[^a-z0-9]', '', lnum)
            if lnum.isdigit():
                label.append(lnum.zfill(4))
            else:
                label.append(lnum.zfill(5))
            file_name.append(filename)

label_list = pd.DataFrame(list(zip(file_name, label)), columns=['file_name','label'])
label_list['label2'] = label_list['label'].str.lstrip('0')
label_list_cut = label_list[['file_name','label2']]
#print(label_list)
label_list_cut.to_csv(folder_path + '/label_list.csv', encoding='utf-8', index=False)
with pd.ExcelWriter(folder_path + '/label_list.xlsx', engine='openpyxl') as writer:
    label_list_cut.to_excel(writer, sheet_name='labels', index=False, startrow=0)
