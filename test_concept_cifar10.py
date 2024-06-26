
import torch
import clip
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
import os
import glob


device = "cuda" if torch.cuda.is_available() else "cpu"
print ("ready to load")
model, preprocess = clip.load("ViT-B/32", device=device)
data_path = os.getcwd() + '/data/cifar10/RC/rc_concept/' ## RED CAR folder

print (data_path)

all_probs = []
all_true_y = []
lst = os.listdir(data_path) # your directory path
print (lst)
concepts = len(lst)
print (concepts)

labels = ["redcar", "non-redcar"]
# labels = ["red", "non-red"]

# labels = ["white", "non-white"]
# labels = ["white cat", "non-white cat"]

# labels = ["front pose", "non-front pose"]
# labels = ["front pose horse", "non-front pose horse"]

print (labels)

probss = []
for i in range(concepts):
    image = preprocess(Image.open(data_path+lst[i])).unsqueeze(0).to(device)
    # text = clip.tokenize().to(device)
    text = clip.tokenize(labels).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # print (type(probs))
        # print (probs[0].tolist())
        probss+=[probs[0][0].tolist()]  
print (len(probss))
all_probs.extend(probss)

probss = []
data_path = os.getcwd() + '/data/cifar10/RC/rc_negconcept/'

lst = os.listdir(data_path) # your directory path
nonconcepts =  len(lst)
print (data_path)

for i in range(nonconcepts):
    image = preprocess(Image.open(data_path+lst[i])).unsqueeze(0).to(device)
    # text = clip.tokenize(["front pose", "non-front pose"]).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        probss+=[probs[0][0].tolist()]  
print (len(probss))

all_probs.extend(probss)

concept_y =  [1] * concepts ## RED CAR
nonconcept_y =  [0] * nonconcepts ## NONREDCAR
all_true_y.extend(concept_y)
all_true_y.extend(nonconcept_y)
print (len(all_true_y))
print (len(all_probs))
print (f'ROC SCORE {roc_auc_score(all_true_y, all_probs)}')

