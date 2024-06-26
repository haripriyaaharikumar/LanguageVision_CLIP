
## conda environment : source activate env_clip
import torch
import clip
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
import os
from torch.utils.data import DataLoader
from all_loader_default_v2 import get_dataset_manager
import glob


device = "cuda" if torch.cuda.is_available() else "cpu"
print ("ready to load")
model, preprocess = clip.load("ViT-B/32", device=device)

# concept = ['painted elephant','unpainted elephant']
# concept = ['wrinkled elephant','non-wrinkled elephant']

# concept = ['ancient library','modern library']
# concept = ['modern library','ancient library']
# concept = ['old library','young library']
# concept = ['empty library', 'full library']

# concept = ['bright lightning', 'dark lightning' ]
concept = ['dark lightning','bright lightning']

def file_read(filename):
    with open(filename, 'r') as f:
                pairs = f.read().strip().split('\n')
    return pairs
data_root = '/home/haripriyak/PycharmProjects/Data/mit-states/images/'


# labels = ['painted','unpainted']
# labels = ['painted elephant','unpainted elephant']
# labels = ['wrinkled','non-wrinkled']
# labels = ['wrinkled elephant','non-wrinkled elephant']

# labels = ['young','old']
# labels = ['ancient','modern']
# labels = ['modern','ancient']
# labels = ['empty','full']
# labels = ['empty library', 'full library']
# labels = ['old','young']
# labels = ['bright', 'dark']
# labels = ['bright lightning', 'dark lightning' ]

# labels = ['dark','bright']

labels = ['dark lightning','bright lightning']


print (concept)
pos_img_paths = file_read(f'./data_ind_file/mit-states/{concept[0]}/pos_test_data.txt')

all_probs = []
all_true_y = []
probss = []
for i in range(len(pos_img_paths)):
    img_pth, lbl = pos_img_paths[i].split('^')
    image = preprocess(Image.open(data_root+img_pth)).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)      
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # print (probs)
        probss+=[probs[0][0].tolist()]  

all_probs.extend(probss)
probss = []
neg_img_paths = file_read(f'./data_ind_file/mit-states/{concept[0]}/neg_test_data.txt')
for i in range(len(neg_img_paths)):
    img_pth, lbl = neg_img_paths[i].split('^')
    image = preprocess(Image.open(data_root+img_pth)).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text) 
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        probss+=[probs[0][0].tolist()]  
        # print (probs)

all_probs.extend(probss)
# print (all_probs)

concept_y =  [1] * len(pos_img_paths) 
nonconcept_y =  [0] * len(neg_img_paths) 
all_true_y.extend(concept_y)
all_true_y.extend(nonconcept_y)

print (f'ROC SCORE {roc_auc_score(all_true_y, all_probs)}')

