from PIL import Image
import numpy as np
import pickle
import os

def get_br_data():
    path = 'data/bob_ross/images/'
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    labels_path = 'data/bob_ross/labels/'
    X = []
    y = []
    for f in files:
        img_path = path + f
        label_path = labels_path + f
        imgsz = (320, 448)
        img = Image.open(img_path).convert('RGB').resize(imgsz)
        img = np.array(img)
        label = Image.open(label_path).convert('RGB').resize(imgsz)
        label = np.array(label)
        X.append(img)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    n = len(y)
    perm = np.random.permutation(range(n))
    train_end = int(0.9 * n)
    X_train = X[perm[:train_end]]
    y_train = y[perm[:train_end]]
    X_val = X[perm[train_end:]]
    y_val = y[perm[train_end:]]
    return (X_train, y_train, X_val, y_val)

def get_br_pickle():
    (X_train, y_train, X_val, y_val) = get_br_data()
    pickle.dump((X_train, y_train, X_val, y_val), open('br.pickle', 'wb+'))
    (X_train, y_train, X_val, y_val) = pickle.load(open('br.pickle', 'rb+'))
    print(X_train.shape)
    print(y_val.shape)

#get_br_pickle()

def get_coco_data():
    path = 'data/val/'
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    labels_path = 'data/annotations/stuff_val2017_pixelmaps/'
    X = np.zeros((5000, 224, 224, 3))
    y = np.zeros((5000, 224, 224, 91))
    colors = {}
    color_no = 0
    crpshp = (0, 0, 224, 224)
    print(len(files))
    for i,f in enumerate(files):
        print(i)
        img_path = path + f
        label_path = labels_path + f
        img = Image.open(img_path).convert('RGB').crop(crpshp)
        img = np.array(img)
        label = Image.open(label_path[:-3] + 'png').convert('RGB').crop(crpshp)
        label = np.array(label)
        H, W, C = label.shape
        new_label = np.zeros((H, W, 91))
        for h in range(H):
            for w in range(W):
                color = tuple(label[h,w])
                if not color in colors:
                    colors[color] = color_no
                    new_label[h,w,colors[color]] = 1
                    color_no += 1
                else:
                    new_label[h,w,colors[color]] = 1
        X[i] = img
        y[i] = new_label
    print(X.shape)
    print(y.shape)
    pickle.dump((X,y), open('coco_val.pickle', 'wb+'))


get_coco_data()
