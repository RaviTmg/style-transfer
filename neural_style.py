from vgg import build_model
import tensorflow as tf
import numpy as np 
import scipy.io  
import argparse 
import struct
import errno
import time                       
import cv2
import os
import glob

MAX_SIZE = 512
content_weight = 5e0
style_weight = 1e4
tv_weight = 1e-3
temporal_weight = 2e2
content_loss_function = 1

original_colors = False

style_imgs_weights = [1.0]

content_layers = ['conv4_2']
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

content_layer_weights = [1.0]
style_layer_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

color_convert_type = 'yuv'
style_mask_imgs = None

seed = 0

optimizer_type = 'lbfgs'
learning_rate = 1e0

max_iterations = 1000
print_iterations = 50

def normalize(weights):
  denom = sum(weights)
  if denom > 0.:
    return [float(i) / denom for i in weights]
  else: return [0.] * len(weights)
style_layer_weights   = normalize(style_layer_weights)
content_layer_weights = normalize(content_layer_weights)
style_imgs_weights    = normalize(style_imgs_weights)

'''
'a neural algorithm for artistic style' loss functions
'''
def content_layer_loss(p, x):
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    if content_loss_function   == 1:
        K = 1. / (2. * N**0.5 * M**0.5)
    elif content_loss_function == 2:
        K = 1. / (N * M)
    elif content_loss_function == 3:  
        K = 1. / 2.
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss

def style_layer_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G

def mask_style_layer(a, x, mask_img):
    _, h, w, d = a.get_shape()
    mask = get_mask_image(mask_img, w.value, h.value)
    mask = tf.convert_to_tensor(mask)
    tensors = []
    for _ in range(d.value): 
        tensors.append(mask)
    mask = tf.stack(tensors, axis=2)
    mask = tf.stack(mask, axis=0)
    mask = tf.expand_dims(mask, 0)
    a = tf.multiply(a, mask)
    x = tf.multiply(x, mask)
    return a, x

def sum_masked_style_losses(sess, net, style_imgs):
    total_style_loss = 0.
    weights = style_imgs_weights
    masks = style_mask_imgs
    for img, img_weight, img_mask in zip(style_imgs, weights, masks):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(style_layers, style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            a, x = mask_style_layer(a, x, img_mask)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss

def sum_style_losses(sess, net, style_imgs):
    total_style_loss = 0.
    weights = style_imgs_weights
    for img, img_weight in zip(style_imgs, weights):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(style_layers, style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss

def sum_content_losses(sess, net, content_img):
    sess.run(net['input'].assign(content_img))
    content_loss = 0.
    for layer, weight in zip(content_layers, content_layer_weights):
        p = sess.run(net[layer])
        x = net[layer]
        p = tf.convert_to_tensor(p)
        content_loss += content_layer_loss(p, x) * weight
    content_loss /= float(len(content_layers))
    return content_loss

'''
  utilities and i/o
'''
def read_image(path):
  # bgr image
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  check_image(img, path)
  img = img.astype(np.float32)
  img = preprocess(img)
  return img

def write_image(path, img):
  img = postprocess(img)
  print('writing image to: ', path)
  cv2.imwrite(path, img)

def preprocess(img):
  imgpre = np.copy(img)
  # bgr to rgb
  imgpre = imgpre[...,::-1]
  # shape (h, w, d) to (1, h, w, d)
  imgpre = imgpre[np.newaxis,:,:,:]
  imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  return imgpre

def postprocess(img):
  imgpost = np.copy(img)
  imgpost += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  # shape (1, h, w, d) to (h, w, d)
  imgpost = imgpost[0]
  imgpost = np.clip(imgpost, 0, 255).astype('uint8')
  # rgb to bgr
  imgpost = imgpost[...,::-1]
  return imgpost

def read_flow_file(path):
  with open(path, 'rb') as f:
    # 4 bytes header
    header = struct.unpack('4s', f.read(4))[0]
    # 4 bytes width, height    
    w = struct.unpack('i', f.read(4))[0]
    h = struct.unpack('i', f.read(4))[0]   
    flow = np.ndarray((2, h, w), dtype=np.float32)
    for y in range(h):
      for x in range(w):
        flow[0,y,x] = struct.unpack('f', f.read(4))[0]
        flow[1,y,x] = struct.unpack('f', f.read(4))[0]
  return flow

def read_weights_file(path):
  lines = open(path).readlines()
  header = list(map(int, lines[0].split(' ')))
  w = header[0]
  h = header[1]
  vals = np.zeros((h, w), dtype=np.float32)
  for i in range(1, len(lines)):
    line = lines[i].rstrip().split(' ')
    vals[i-1] = np.array(list(map(np.float32, line)))
    vals[i-1] = list(map(lambda x: 0. if x < 255. else 1., vals[i-1]))
  # expand to 3 channels
  weights = np.dstack([vals.astype(np.float32)] * 3)
  return weights


def maybe_make_directory(dir_path):
  if not os.path.exists(dir_path):  
    os.makedirs(dir_path)

def check_image(img, path):
  if img is None:
    raise OSError(errno.ENOENT, "No such file", path)

    '''
    rendering -- where the magic happens
    '''
def stylize(content_img, style_imgs, init_img, frame=None, device='/gpu:0', style_mask=False):
    with tf.device(device), tf.Session() as sess:
        # setup network
        net = build_model(content_img)
        
        # style loss
        if style_mask:
            L_style = sum_masked_style_losses(sess, net, style_imgs)
        else:
            L_style = sum_style_losses(sess, net, style_imgs)
        
        # content loss
        L_content = sum_content_losses(sess, net, content_img)
        
        # denoising loss
        L_tv = tf.image.total_variation(net['input'])
        
        # loss weights
        alpha = content_weight
        beta  = style_weight
        theta = tv_weight
        
        # total loss
        L_total  = alpha * L_content
        L_total += beta  * L_style
        L_total += theta * L_tv
        
        # optimization algorithm
        optimize_fn = get_optimizer('lbfgs', L_total)

        if optimizer_type == 'adam':
            minimize_with_adam(sess, net, optimize_fn, init_img, L_total)
        elif optimizer_type == 'lbfgs':
            minimize_with_lbfgs(sess, net, optimize_fn, init_img)
        
        output_img = sess.run(net['input'])
        
        if original_colors:
            output_img = convert_to_original_colors(np.copy(content_img), output_img)
        sess.close()
    return output_img

def minimize_with_lbfgs(sess, net, optimizer, init_img):
    print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)

def minimize_with_adam(sess, net, optimizer, init_img, loss):
    print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
    train_op = optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations = 0
    while (iterations < max_iterations):
        sess.run(train_op)
        if iterations % print_iterations == 0:
            curr_loss = loss.eval()
            print("At iteration {}\tf=  {}".format(iterations, curr_loss))
        iterations += 1

def get_optimizer(optimizer_type, loss):
    if optimizer_type == 'lbfgs':
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
        loss, method='L-BFGS-B',
        options={'maxiter': max_iterations,
                    'disp': print_iterations})
    elif optimizer_type == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer


'''
image loading and processing
'''
def get_init_image(init_type, content_img, style_imgs, frame=None, noise_ratio=1.0):
    if init_type == 'content':
        return content_img
    elif init_type == 'style':
        return style_imgs[0]
    elif init_type == 'random':
        init_img = get_noise_image(noise_ratio, content_img)
        return init_img

def get_content_image(content_img):
    # bgr image
    img = cv2.imread(content_img, cv2.IMREAD_COLOR)
    check_image(img, content_img)
    img = img.astype(np.float32)
    h, w, d = img.shape
    mx = MAX_SIZE
    # resize if > max size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    return img

def get_style_images(style_imgs, shape):
    _, ch, cw, cd = shape
    style_images = []
    for style_fn in style_imgs:
        # bgr image
        img = cv2.imread(style_fn, cv2.IMREAD_COLOR)

        check_image(img, style_fn)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
        img = preprocess(img)
        style_images.append(img)
    return style_images

def get_noise_image(noise_ratio, content_img):
    np.random.seed(seed)
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    img = noise_ratio * noise_img + (1.-noise_ratio) * content_img
    return img

def get_mask_image(mask_img, width, height):
    img = cv2.imread(mask_img, cv2.IMREAD_GRAYSCALE)
    check_image(img, mask_img)
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    mx = np.amax(img)
    img /= mx
    return img

def warp_image(src, flow):
    _, h, w = flow.shape
    flow_map = np.zeros(flow.shape, dtype=np.float32)
    for y in range(h):
        flow_map[1,y,:] = float(y) + flow[1,y,:]
    for x in range(w):
        flow_map[0,:,x] = float(x) + flow[0,:,x]
    # remap pixels to optical flow
    dst = cv2.remap(
        src, flow_map[0], flow_map[1], 
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    return dst

def convert_to_original_colors(content_img, stylized_img):
    content_img  = postprocess(content_img)
    stylized_img = postprocess(stylized_img)
    if color_convert_type == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
    elif color_convert_type == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif color_convert_type == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
    elif color_convert_type == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    dst = preprocess(dst)
    return dst

def render_single_image(content_image, style_image, init_img_type, save_dir):
    maybe_make_directory(save_dir)

    _, cont_file = os.path.split(content_image)
    cont_file = cont_file.split('.')[0]

    content_img = get_content_image(content_image)
    style_imgs = get_style_images([style_image], content_img.shape)
    
    with tf.Graph().as_default():
        
        init_img = get_init_image(init_img_type, content_img, style_imgs)
        tick = time.time()
        output_image = stylize(content_img, style_imgs, init_img)
        write_image(os.path.join(save_dir, cont_file+'-content.png'), content_img)
        write_image(os.path.join(save_dir, cont_file+'-result.png'), output_image)
        tock = time.time()
        print('Single image elapsed time: {}'.format(tock - tick))


if __name__ == "__main__":
    contents = glob.glob('data/content/*.png')
    for idx, image in enumerate(contents[0:10]):
        print('RENDERING {}th IMAGE: {}'.format( idx, image))
        render_single_image(image, 'data/style/Bole_Sixote.1.png', 'content', 'Bole_Sixote.2')
