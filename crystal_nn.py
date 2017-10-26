#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from multiprocessing import Process
from multiprocessing import Queue

import chainer
import numpy as np
import os
import shutil
import six
import sys
import time
import itertools
import h5py


from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from mendeleev import element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


all_elements = [Element.from_Z(z).symbol for z in range(1,104)]
masses = dict(zip(all_elements,[element(e).atomic_weight for e in all_elements]))
radii = dict(zip(all_elements,[element(e).covalent_radius_slater for e in all_elements]))
radii['Pb'] = 180.
en_paul = dict(zip(all_elements,[element(e).en_pauling for e in all_elements]))

ATTRIBUTES_KEY = {
'atomic_number': lambda e: e.Z if e.Z != 0 else 'failed',
'electronegativity': lambda e: en_paul[e.symbol] if en_paul[e.symbol] != None else 'failed',
'row': lambda e: e.row,
'group': lambda e: e.group,
'is_noble_gas': lambda e: int(e.is_noble_gas),
'is_transition_metal': lambda e: int(e.is_transition_metal),
'is_rare_earth_metal': lambda e: int(e.is_rare_earth_metal),
'is_metalloid': lambda e: int(e.is_metalloid),
'is_alkali': lambda e: int(e.is_alkali),
'is_alkaline': lambda e: int(e.is_alkaline),
'is_halogen': lambda e: int(e.is_halogen),
'is_lanthanoid': lambda e: int(e.is_lanthanoid),
'is_actinoid': lambda e: int(e.is_actinoid),
'atomic_mass': lambda e: masses[e.symbol] if (masses[e.symbol] != None and masses[e.symbol] != 0) else 'failed',
'atomic_radius': lambda e: radii[e.symbol] if (radii[e.symbol] != None and radii[e.symbol] != 0) else 'failed'
}

def element_features(element):
  return np.array([
    ATTRIBUTES_KEY['atomic_mass'](element),
    ATTRIBUTES_KEY['atomic_radius'](element),
    ATTRIBUTES_KEY['electronegativity'](element)
    ])

def make_crystal_img(cif_str, resolution=0.5):
  struct = Structure.from_str(input_string=cif_str,fmt='cif')
  sg = SpacegroupAnalyzer(struct)
  conv_struct = sg.get_conventional_standard_structure()
  sites = conv_struct.sites

  nfeats = [3]
  lattice_lengths = np.asarray(conv_struct.lattice.abc)
  dimensions = list(np.ceil(lattice_lengths / resolution).astype(int))
  img = np.zeros(dimensions+nfeats, dtype=np.float32)

  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      for k in range(img.shape[2]):
        x = (float(i)+0.5)/img.shape[0]*lattice_lengths[0]
        y = (float(j)+0.5)/img.shape[1]*lattice_lengths[1]
        z = (float(k)+0.5)/img.shape[2]*lattice_lengths[2]
        pt = np.array([x,y,z])
        neighbors = conv_struct.get_sites_in_sphere(pt=pt,r=10)
        nearest_neighbor = [n[0] for n in sorted(neighbors, key=lambda s: s[1])][0]
        features = element_features(nearest_neighbor.specie)
        if all([f != 'failed' for f in features]):
          img[i,j,k,:] = features
        else:
          return 'failed'
  return img

def offset_img(img, offset=None):
  if offset==None:
    xoffset = np.random.randint(0,img.shape[0])
    yoffset = np.random.randint(0,img.shape[1])
    zoffset = np.random.randint(0,img.shape[2])
  else:
    xoffset = offset[0]
    yoffset = offset[1]
    zoffset = offset[2]

  return np.roll(img,(xoffset,yoffset,zoffset),(0,1,2))

def rotate_img(img, rotation=None):
  if rotation == None:
    rotation = np.random.randint(0,4,3)
  img = np.rot90(img,rotation[0],(0,1))
  img = np.rot90(img,rotation[1],(1,2))
  img = np.rot90(img,rotation[2],(2,0))
  return img

def tile_img(img, out_shape):
  reps = np.ceil(np.asarray(out_shape)/np.asarray(img.shape)).astype(int)
  tiled_img = np.tile(img,reps)
  out_img = tiled_img[:out_shape[0],:out_shape[1],:out_shape[2],:]
  return out_img

def load_data(data_file):
  data = h5py.File(data_file,'r')

  labels = []
  imgs = []
  for im in data['Unstable'].keys():
    labels.append(np.array([0,1]))
    imgs.append(data['Unstable'][im])
  for im in data['Stable'].keys():
    labels.append(np.array([1,0]))
    imgs.append(data['Stable'][im])

  return data, labels

def one_epoch(parameters,model,optimizer,img_data,img_labels,epoch,train):
  model.train = train

  xp = cuda.cupy if parameters['gpu'] >= 0 else np
  GPU_on = True if parameters['gpu'] >= 0 else False

  L_Y_train = len(img_labels)
  time1 = time.time()

  np.random.seed(int(time.time()))
  perm = np.random.permutation(L_Y_train)

  sum_accuracy = 0
  sum_loss = 0
  num = 0
  for i in range(0,L_Y_train,parameters['batchsize']):
    if train:
      img_batch = [rotate_img(offset_img(img_data[idx])) for idx in perm[i:i+parameters['batchsize']]
      tiled_batch = [tile_img(im, parameters['input_dims']) for im in img_batch]
      label_batch = [img_labels[idx] for idx in perm[i:i+parameters['batchsize']]]

    if (GPU_on):
      tiled_batch = cuda.to_gpu(data_batch)
      label_batch = cuda.to_gpu(label_batch)

    # convert to chainer variable
    x = Variable(xp.asarray(tiled_batch))
    t = Variable(xp.asarray(label_batch))

    model.zerograds()

    if train:
      optimizer.update(model, x, t)

      if epoch == 1 and num == 0:
        with open('{}/graph.dot'.format(parameters['result_dir']), 'w') as o:
          g = computational_graph.build_computational_graph(
            (model.loss, ), remove_split=True)
          o.write(g.dump())
      sum_loss += float(model.loss.data) * len(t.data)
      sum_accuracy += float(model.accuracy.data) * len(t.data)
      num += t.data.shape[0]
      logging.info('{:05d}/{:05d}\tloss:{:.3f}\tacc:{:.3f}'.format(
        num, data.shape[0], sum_loss / num, sum_accuracy / num))
    else:
      pred = model(x, t).data
      sum_accuracy +=  float(sum(pred.argmax(axis=1) == t.data))
      num += t.data.shape[0]
    del x,t

  if train and (epoch == 1 or epoch % parameters['snapshot'] == 0):
    model_fn = '{}/epoch-{}.model'.format(parameters['result_dir'], epoch)
    opt_fn = '{}/epoch-{}.state'.format(parameters['result_dir'], epoch)
    serializers.save_hdf5(model_fn, model)
    serializers.save_hdf5(opt_fn, optimizer)

  if train:
    logging.info('epoch:{}\ttrain loss:{:.4f}\ttrain accuracy:{:.4f}'.format(
      epoch, sum_loss / num, sum_accuracy / num))
  else:
    logging.info('epoch:%d\ttest accuracy:%0.4f'%(epoch,sum_accuracy/num))

def get_model_optimizer(parameters):
    model_path = os.path.join(os.getcwd(),'Models',parameters['model_name']+'.py')
    model = imp.load_source(parameters['model_name'], model_path).model

    dst = '%s/%s' % (parameters['result_dir'], model_path)
    if not os.path.exists(dst):
      shutil.copy(model_path, dst)

    dst = '%s/%s' % (parameters['result_dir'], os.path.basename(__file__))
    if not os.path.exists(dst):
      shutil.copy(__file__, dst)

    # prepare model
    if parameters['gpu'] >= 0:
      cuda.get_device(parameters['gpu']).use()
      model.to_gpu()

    # prepare optimizer
    if parameters['optimizer'] == 'MomentumSGD':
      optimizer = optimizers.MomentumSGD(lr=parameters['learning_rate'], momentum=parameters['momentum'])
    elif parameters['optimizer'] == 'Adam':
      optimizer = optimizers.Adam(alpha=parameters['alpha'])
    elif parameters['optimizer'] == 'AdaGrad':
      optimizer = optimizers.AdaGrad(lr=parameters['learning_rate'])
    elif parameters['optimizer'] == 'RMSprop':
      optimizer = optimizers.RMSprop(lr=parameters['learning_rate'],alpha=parameters['alpha'])
    else:
      raise Exception('No optimizer is selected')

    optimizer.setup(model)

    if args.opt == 'MomentumSGD':
      optimizer.add_hook(
        chainer.optimizer.WeightDecay(parameters['weight_decay']))
    return model, optimizer


if __name__ == '__main__':
  # Parameters
  parameters = {
  'data_file': 'SpinelData_0.5A.hdf5',
  'model_name': 'TestModel',
  'optimizer': 'MomentumSGD',
  'input_dims': [32,32,32],
  'train_test_split': 0.1,
  'learning_rate': 0.1,
  'decay_ratio': 0.1,
  'decay_freq': 1,
  'momentum': 0.9,
  'weight_decay': 0.0001,
  'alpha': 0.001,
  'num_epochs': 200,
  'batchsize': 64,
  'gpu': 0,
  'snapshot': 10,
  'validate_freq': 1
  }
  
  # Set up logging
  if len(sys.argv) > 1:
    dir_name = sys.argv[2]
  else:
    dir_name = parameters['model_name']
  dir_name = 'Results/%s_%s'%(dir_name,time.asctime())

  parameters['result_dir'] = os.path.join(os.getcwd(),dir_name)
  os.makedirs(parameters['result_dir'])
  logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename="%s/log.txt"%parameters['result_dir'], level=logging.DEBUG)

  # Create model and optimizer
  model, optimizer = get_model_optimizer(args)

  # Load data
  imgs, labels = load_data(parameters['data_file'])

  # Randomize and split train / test
  np.random.seed(int(time.time()))
  nimgs = len(labels)
  perm = np.random.permutation(nimgs)
  test_idxs = perm[:nimgs*parameters['train_test_split']]
  train_idxs = perm[nimgs*parameters['train_test_split']:]

  train_imgs = [imgs[i] for i in train_idxs]
  train_labels = [labels[i] for i in train_idxs]
  test_imgs = [imgs[i] for i in test_idxs]
  test_labels = [labels[i] for i in test_idxs]

  # Learning
  for epoch in range(1, parameters['num_epochs'] + 1):
    logging.info('learning rate:{}'.format(optimizer.lr))
    one_epoch(parameters,model,optimizer,train_imgs,train_labels,epoch,True)

    if epoch == 1 or epoch % parameters['validate_freq'] == 0:
      one_epoch(parameters, model, optimizer, test_imgs, test_labels, epoch, False)

    if parameters['optimizer'] == 'MomentumSGD' and epoch % parameters['decay_freq'] == 0:
      optimizer.lr *= parameters['decay_ratio']
