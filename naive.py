# Naive segmentation algorithm based on region growing and shrinking
import os
import numpy as np
from PIL import Image


def get_seeds(H, W, n):
    hs = np.random.choice(range(H), size=(n,))
    ws = np.random.choice(range(W), size=(n,))
    out = []
    for i in range(n):
        out.append((hs[i], ws[i]))
    return out


def get_adjacent(pixel, H, W, visited):
    x = pixel[0]
    y = pixel[1]
    out = []
    for a in range(x-1, x+2):
        for b in range(y-1, y+2):
            if not (a == x and b == y):
                if not (a > x or a < 0 or b < 0 or b > y):
                    coord = (a,b)
                    if not coord in visited:
                        out.append(coord)
    return out


def is_similar(color1, color2, eps=320):
    return np.linalg.norm(color1 - color2) < eps


def prune(sets):
    merged = True
    while merged:
        merged = False
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    return sets


def get_regions(img, seeds):
    H, W, C = img.shape
    new_img = np.zeros(img.shape)
    colors = [img[seed[0], seed[1]] for seed in seeds]
    regions = []
    for i,seed in enumerate(seeds):
        region = set()
        color = img[seed[0], seed[1]]
        neighbors = [(seed[0], seed[1])]
        visited = set()
        while len(neighbors) > 0:
            here = neighbors.pop()
            region.add(here)
            adjacent_pixels = get_adjacent(here, H, W, visited)
            for adj in adjacent_pixels:
                visited.add(adj)
                adj_color = img[adj[0], adj[1]]
                if is_similar(color, adj_color):
                    neighbors.append(adj)
        regions.append(region)
    regions = prune(regions)
    return regions


def get_cluster(coord, regions):
    for i,region in enumerate(regions):
        if coord in region:
            return i
    return -1


def get_new_img(img, regions):
    colors = np.random.choice(range(256), size=(len(regions),3))
    new_img = np.zeros(img.shape)
    H, W, C = img.shape
    for h in range(H):
        for w in range(W):
            cluster = get_cluster((h, w), regions)
            if not cluster == -1:
                new_img[h,w,:] = colors[cluster,:]
    return new_img


def segment_image(filename):
    img = Image.open(filename).convert('RGB')
    img = np.array(img)
    H, W, C = img.shape
    n = 20
    seeds = get_seeds(H, W, n)
    regions = get_regions(img, seeds)
    new_img = get_new_img(img, regions)
    return new_img


def convert_label(image):
    H, W, C = image.shape
    pixels = {}
    i = 0
    for h in range(H):
        for w in range(W):
            pixel = tuple(image[h,w])
            if not pixel in pixels:
                pixels[pixel] = i
                i += 1
    new_image = np.zeros((H,W,1))
    for h in range(H):
        for w in range(W):
            pixel = tuple(image[h,w])
            new_image[h,w] = pixels[pixel]
    return new_image


def convert_back(label):
    H, W, _ = label.shape
    new = np.zeros((H,W,3))
    for h in range(H):
        for w in range(W):
            new[h, w, :] = label[h,w]
    return new

def evaluate_algo():
    path = 'data/bob_ross/images/'
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    labels_path = 'data/bob_ross/labels/'
    mse = 0
    min_mse = float('inf')
    for name in files:
        filename = path + name
        segmented = segment_image(filename)
        segmented = convert_label(segmented)
        labelname = labels_path + name
        label = Image.open(labelname).convert('RGB')
        label = np.array(label)
        label = convert_label(label)
        curr_mse = np.mean((segmented-label)**2)
        if curr_mse < min_mse:
            min_mse = curr_mse
            segmented = convert_back(segmented)
            im = Image.fromarray(segmented.astype(np.uint8))
            im.save("br_segmented.png")
            label = convert_back(label)
            im = Image.fromarray(label.astype(np.uint8))
            im.save("br_label.png")
        mse += curr_mse
    print(f"MSE: {mse}")



if __name__ == '__main__':
    evaluate_algo()
    #segment_image('pic.jpg')
