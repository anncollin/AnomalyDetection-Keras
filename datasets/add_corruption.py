from utils.import_lib import *
from utils.helper import *

from random import randint, uniform
from numpy.random import normal, uniform
from skimage.util import random_noise
from skimage.filters import threshold_otsu
from skimage.draw import ellipse_perimeter, circle
from scipy.interpolate import interp1d 
from scipy.ndimage import gaussian_filter

""" COLLIN Anne-Sophie """ 

""" -----------------------------------------------------------------------------------------
Corrupts a single with some type of noise
INPUT : 
    - img: image to corrupt
    - args: dictionnary with all arguments (built with the GUI)
OUTPUT: 
    - temp: corrupted image
----------------------------------------------------------------------------------------- """ 
def corrupt_image(img, args): 
    default = args['1_default']
    if default == ['Gaussian'] :
        img_modified = add_gaussian(img, float(args['1_p_0']))
    elif default == 'Gaussian' :
        img_modified = add_gaussian(img, float(args['std']))

    elif default == ['Stain'] :
        img_modified = add_stain(img, args['1_p_0'], args['1_p_1'], float(args['1_p_2']), float(args['1_p_3']))
    elif default == 'Stain' :
        img_modified = add_stain(img, args['size'], args['color'], float(args['irregularity']), float(args['blur']))


    elif default == ['Scratch'] : 
        img_modified = add_scratch(img, args['1_p_0'])
    elif default == 'Scratch' : 
        img_modified = add_scratch(img, args['color'])


    elif default == ['Drops'] :
        img_modified = add_drops(img, args['1_p_0'], args['1_p_1'], float(args['1_p_2']), int(args['1_p_3']), float(args['1_p_3']))
    elif default == 'Drops' :
        img_modified = add_drops(img, args['size'], args['color'], float(args['irregularity']), int(args['number']), float(args['spacing']))


    elif default == ['Structural'] : 
        prob = uniform(0,1)
        if prob < float(args['1_p_0']) :
            img_modified = add_stain(img, "1-12", "10-220", 0.1, 0.15)
        elif prob >= float(args['1_p_0']) and prob <= ( float(args['1_p_0']) + float(args['1_p_1'])): 
            img_modified = add_scratch(img,"10-220")
        else : 
            img_modified = add_drops(img, "1-2", "10-220", 0.1, 10, 0.1)
    elif default == 'Structural' : 
        prob = uniform(0,1)
        if prob < float(args['drops']) :
            img_modified = add_stain(img, "1-12", "10-220", 0.1, 0.15)
        elif prob >= float(args['drops']) and prob <= ( float(args['drops']) + float(args['scrath'])): 
            img_modified = add_scratch(img,"10-220")
        else : 
            img_modified = add_drops(img, "1-2", "10-220", 0.1, 10, 0.1)

    elif default == ['StainOrGaussian'] : 
        prob = uniform(0,1)
        if prob < float(args['1_p_0']) :
            img_modified = add_stain(img, "1-12", "10-220", 0.1, 0.15)
        else : 
            img_modified = add_gaussian(img, 0.1)
    elif default == 'StainOrGaussian' : 
        prob = uniform(0,1)
        if prob < float(args['stain']) :
            img_modified = add_stain(img, "1-12", "10-220", 0.1, 0.15)
        else : 
            img_modified = add_gaussian(img, 0.1)

    else : 
        img_modified = img
    return img_modified

""" -----------------------------------------------------------------------------------------
Corrupts an array of images with some type of noise
INPUT : 
    - im_array: array of images to corrupt
    - args: dictionnary with all arguments
OUTPUT: 
    - temp: corrupted array of images
----------------------------------------------------------------------------------------- """ 
def corrupt(im_array, args):
    temp = np.copy(im_array)
    for idx,img in enumerate(np.copy(im_array)):
        temp[idx] = corrupt_image(img, args)
    return temp


""" -----------------------------------------------------------------------------------------
Corrupts the training set of a dataset object 
INPUT : 
    - dataset: a dataset object
    - path: path to the configuration file (json)
OUTPUT: 
    - temp: corrupted array of images
----------------------------------------------------------------------------------------- """ 
def corrupt_dataset(im_array, path):
    params  = read_json(path)
    temp = np.copy(im_array)
    for idx,img in enumerate(np.copy(im_array)):
        temp[idx] = corrupt_image(img, params)
    return temp

""" -----------------------------------------------------------------------------------------
Draw an ellipse like shape 
INPUT : 
    - img: image to corrupt with an elliptical stain
    - size: size of the ellipse in terms of % of image row/col
    - color: mean pixel intensity of the added noise
    - irregularity: level of irregularity of the ellipse
    - blur: blur edges (0: no)
OUTPUT: 
    - corrupted image
----------------------------------------------------------------------------------------- """ 
def add_stain(img, size, color, irregularity, blur):

    if '-' not in color: 
        color = int(color)
    else: 
        min_color, max_color = int(color.split('-')[0]), int(color.split('-')[1])
        color                = randint(min_color, max_color)
    col, row             = img.shape[1], img.shape[0]
    min_range, max_range = float(size.split('-')[0]), float(size.split('-')[1])
    a, b                 = randint(int(min_range/100.*col), int(max_range/100.*col)), randint(int(min_range/100.*row), int(max_range/100.*row))
    rotation             = uniform(0, 2*np.pi)

    cx, cy   = randint(max(a,b), int(col-max(a,b))), randint(max(a,b), int(row-max(a,b)))
    x,y      = ellipse_perimeter(cy, cx, a, b, rotation)
    contour  = np.array([[i,j] for i,j in zip(x,y)])

    # Change the shape of the ellipse 
    if irregularity > 0: 
        contour = perturbate_ellipse(contour, cx, cy, (a+b)/2, irregularity)

    mask = np.zeros((row, col)) 
    mask = cv2.drawContours(mask, [contour], -1, 1, -1)

    if blur != 0 : 
        mask = gaussian_filter(mask, max(a,b)*blur)

    if img.shape[-1] == 1: 
        rgb_mask     = np.dstack([mask])
    else:
        rgb_mask     = np.dstack([mask]*3)
    not_modified = np.subtract(np.ones(img.shape), rgb_mask)
    stain        = 255*random_noise(np.zeros(img.shape), mode='gaussian', mean = color/255., var = 0.05/255.)
    result       = np.add( np.multiply(img,not_modified), np.multiply(stain,rgb_mask) ) 

    return result.astype(np.uint8)

def perturbate_ellipse(contour, cx, cy, diag, irregularity):
    # Keep only some points
    if len(contour) < 20: 
        pts = contour
    else: 
        pts = contour[0::int(len(contour)/20)]

    # Perturbate coordinates
    for idx,pt in enumerate(pts): 
        pts[idx] = [pt[0]+randint(-int(diag*irregularity),int(diag*irregularity)), pt[1]+randint(-int(diag*irregularity),int(diag*irregularity))]
    pts = sorted(pts, key=lambda p: clockwiseangle(p, cx, cy))
    pts.append([pts[0][0], pts[0][1]])

    # Interpolate between remaining points
    i = np.arange(len(pts))
    interp_i = np.linspace(0, i.max(), 10 * i.max())
    xi = interp1d(i, np.array(pts)[:,0], kind='cubic')(interp_i)
    yi = interp1d(i, np.array(pts)[:,1], kind='cubic')(interp_i) 
 
    return np.array([[int(i),int(j)] for i,j in zip(yi,xi)])

def clockwiseangle(point, cx, cy):
    refvec = [0 , 1]
    vector = [point[0]-cy, point[1]-cx]
    norm   = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if norm == 0:
        return -math.pi
    normalized = [vector[0]/norm, vector[1]/norm]
    dotprod    = normalized[0]*refvec[0] + normalized[1]*refvec[1] 
    diffprod   = refvec[1]*normalized[0] - refvec[0]*normalized[1] 
    angle      = math.atan2(diffprod, dotprod)
    if angle < 0:
        return 2*math.pi+angle
    return angle


""" -----------------------------------------------------------------------------------------
Draw small circular shapes 
INPUT : 
    - img: image to corrupt with an elliptical stain
    - size: radius of the drops in terms of % of image row/col
    - color: mean pixel intensity of the added noise
    - irregularity: level of irregularity of the ellipse
    - nb: number of drops
    - spacing: deviation around center of the default structure 
    - location: string to describe where the corruption is located 
OUTPUT: 
    - corrupted image
----------------------------------------------------------------------------------------- """ 
def add_drops(img, size, color, irregularity, nb, spacing):
    if '-' not in color: 
        color = int(color)
    else: 
        min_color, max_color = int(color.split('-')[0]), int(color.split('-')[1])
        color = randint(min_color, max_color)
    col, row = img.shape[1], img.shape[0]
    min_range, max_range = float(size.split('-')[0]), float(size.split('-')[1])
    radius = randint(int(min_range/100.*col), int(max_range/100.*col)) +1

    shift = np.log(nb)+5
    cx, cy   = randint(int(shift*radius), int(col-shift*radius)), randint(int(shift*radius), int(row-shift*radius))

    # Print nb circles
    mask = np.zeros((row, col))
    for _ in range(nb):
        ccx, ccy = normal(cx, spacing*radius/10), normal(cy, spacing*radius/10)
        x,y = circle(ccy, ccx, radius)
        new_x, new_y = [], []
        for xx, yy in zip(x, y): 
            if xx < col-1 and yy < row-1: 
                new_x.append(xx)
                new_y.append(yy)
        mask[new_y,new_x] = 1
        
    if img.shape[-1] == 1: 
        rgb_mask     = np.dstack([mask])
    else:
        rgb_mask     = np.dstack([mask]*3)
    not_modified = np.subtract(np.ones(img.shape), rgb_mask)
    stain        = 255*random_noise(np.zeros(img.shape), mode='gaussian', mean = color/255., var = 0.05/255.)
    result       = np.add( np.multiply(img,not_modified), np.multiply(stain,rgb_mask) ) 

    return result.astype(np.uint8)


""" -----------------------------------------------------------------------------------------
Add gaussian noise to the image
INPUT : 
    - img: image to corrupt with gaussian noise
    - sigma: noise standard deviation
OUTPUT: 
    - corrupted image
----------------------------------------------------------------------------------------- """ 
def add_gaussian(img, sigma):
    noise  = 255*random_noise(img, mode='gaussian', mean = 0, var = sigma)
    result = np.add(img,noise)
    return result.astype(np.uint8)

""" -----------------------------------------------------------------------------------------
Add a scratch (on mean intensity) on the image -> a line, a sine wave or a square root 
shaped curve
INPUT : 
    - img: image to corrupt with a scratch
    - color: mean pixel intensity of the added noise
OUTPUT: 
    - corrupted image
----------------------------------------------------------------------------------------- """ 
def add_scratch(img, color):
    if '-' not in color: 
        color = int(color)
    else: 
        min_color, max_color = int(color.split('-')[0]), int(color.split('-')[1])
        color = randint(min_color, max_color)

    max_x, max_y     = img.shape[1], img.shape[0]
    start_point_x    = randint(int(max_x/6), int(max_x - max_x/6 + 1))
    start_point_y    = randint(int(max_y/6), int(max_y - max_y/6 + 1))
    length_scratch_x = randint(start_point_x, max_x)/4
    length_scratch_y = randint(start_point_y, max_y)/4

    scratch_shape     = ['line', 'sin', 'root']
    scratch_direction = ['right', 'left', 'down', 'up']
    shape             = randint(0,2)
    direction         = randint(0,3)

    list_point = []
    if scratch_shape[shape] == 'line':
        list_point = func_x(start_point_x, length_scratch_x, start_point_y, length_scratch_y,
                            scratch_direction[direction], _line, img.shape)
    elif scratch_shape[shape] == 'sin':
        list_point = func_x(start_point_x, length_scratch_x, start_point_y, length_scratch_y,
                            scratch_direction[direction], _sin, img.shape)
    elif scratch_shape[shape] == 'root':
        list_point = func_x(start_point_x, length_scratch_x, start_point_y, length_scratch_y,
                            scratch_direction[direction], _root, img.shape)

    for x,y in list_point:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                point_x, point_y = (x + dx, y + dy)
                if 0 <= point_x < max_x and 0 <= point_y < max_y:
                    new_val = np.random.normal(color, 20, 1)
                    if new_val > 0:
                        img[point_y, point_x] = new_val
                    else:
                        img[point_y, point_x] = 0

    return img

def func_x(start_x, length_x, start_y, length_y, direction, func, shape):
    set_point = list()
    if direction == 'up':
        for x in np.arange(0.0, length_x, 0.1):
            point_x, point_y = (start_x + int(x), start_y + int(func(x)))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))
    elif direction == 'down':
        for x in np.arange(0.0, -1*length_x, -0.1):
            point_x, point_y = (start_x + int(x), start_y + int(func(x)))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))
    elif direction == 'right':
        for y in np.arange(0.0, length_y, 0.1):
            point_x, point_y = (start_x + int(func(y)), start_y + int(y))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))
    elif direction == 'left':
        for y in np.arange(0.0, -1*length_y, -0.1):
            point_x, point_y = (start_x + int(func(y)), start_y + int(y))
            if 0 <= point_x < shape[1] and 0 <= point_y < shape[0]:
                set_point.append((point_x, point_y))

    return set_point

def _line(x):
    return 2*x

def _sin(x):
    return 16*math.sin(math.radians(x*4))

def _root(x):
    return 10*math.sqrt(x) if x >= 0 else 10*math.sqrt(-1*x)



