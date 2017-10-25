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

import argparse
import chainer
import imp
import logging
import numpy as np
import os
import shutil
import six
import sys
import time
import itertools


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

def get_nn_feats(x,y,z):
  pt = np.array([x,y,z])
  neighbors = conv_struct.get_sites_in_sphere(pt=pt,r=5)
  nearest_neighbor = [n[0] for n in sorted(neighbors, key=lambda s: s[1])][0]
  return element_features(nearest_neighbor.specie)

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
        neighbors = conv_struct.get_sites_in_sphere(pt=pt,r=5)
        nearest_neighbor = [n[0] for n in sorted(neighbors, key=lambda s: s[1])][0]
        img[i,j,k,:] = element_features(nearest_neighbor.specie)
  return img


def offset_data(img, offset=None):
  if offset==None:
    xoffset = np.random.randint(0,img.shape[0])
    yoffset = np.random.randint(0,img.shape[1])
    zoffset = np.random.randint(0,img.shape[2])
  else:
    xoffset = offset[0]
    yoffset = offset[1]
    zoffset = offset[2]

  return np.roll(img,(xoffset,yoffset,zoffset),(0,1,2))

def rotate_data(img, rotation=None):
  if rotation == None:
    rotation = np.random.randint(0,4,3)
  img = np.rot90(img,rotation[0],(0,1))
  img = np.rot90(img,rotation[1],(1,2))
  img = np.rot90(img,rotation[2],(2,0))
  return img

def tile_data(img, out_shape):
  reps = np.ceil(np.asarray(out_shape)/np.asarray(img.shape)).astype(int)
  tiled_img = np.tile(img,reps)
  out_img = tiled_img[:out_shape[0],:out_shape[1],:out_shape[2]]
  return out_img

def create_result_dir(args):
  result_dir = 'results/' + os.path.basename(args.model).split('.')[0]
  result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
  result_dir += str(time.time()).replace('.', '')
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
  log_fn = '%s/log.txt' % result_dir
  logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=log_fn, level=logging.DEBUG)
  logging.info(args)

  args.log_fn = log_fn
  args.result_dir = result_dir


def get_model_optimizer(args):
  model_fn = os.path.basename(args.model)
  model = imp.load_source(model_fn.split('.')[0], args.model).model

  dst = '%s/%s' % (args.result_dir, model_fn)
  if not os.path.exists(dst):
    shutil.copy(args.model, dst)

  dst = '%s/%s' % (args.result_dir, os.path.basename(__file__))
  if not os.path.exists(dst):
    shutil.copy(__file__, dst)

  # prepare model
  if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

  # prepare optimizer
  if 'opt' in args:
    # prepare optimizer
    if args.opt == 'MomentumSGD':
      optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    elif args.opt == 'Adam':
      optimizer = optimizers.Adam(alpha=args.alpha)
    elif args.opt == 'AdaGrad':
      optimizer = optimizers.AdaGrad(lr=args.lr)
    else:
      raise Exception('No optimizer is selected')

    optimizer.setup(model)

    if args.opt == 'MomentumSGD':
      optimizer.add_hook(
        chainer.optimizer.WeightDecay(args.weight_decay))
    return model, optimizer
  else:
    print('No optimizer generated.')
    return model


def one_epoch_resnet(args,model,optimizer,cif_data,cif_labels,epoch,train):
  model.train = train

  xp = cuda.cupy if args.gpu >= 0 else np

  
  GPU_on = True if args.gpu >= 0 else False
  # logging.info('data loading started')

  #transfered model to gpu in get_model_optimizer
  #optimizer setup model called in get_model_optimizer

  L_Y_train = len(cif_labels)
  time1 = time.time()

  np.random.seed(int(time.time()))
  perm = np.random.permutation(cif_data.shape[0])
  data = cif_data[perm]
  label = cif_labels[perm]

  sum_accuracy = 0
  sum_loss = 0
  num = 0
  for i in range(0,data.shape[0],args.batchsize):
    data_batch = data[i:i+args.batchsize]
    label_batch =label[i:i+args.batchsize]

    if train:
      data_batch = augment_data(data_batch, random_rotation=True)

    if (GPU_on):
      data_batch = cuda.to_gpu(data_batch)
      label_batch = cuda.to_gpu(label_batch)

    # convert to chainer variable
    x = Variable(xp.asarray(data_batch))
    t = Variable(xp.asarray(label_batch))

    model.zerograds()

    if train:
      optimizer.update(model, x, t)

      if epoch == 1 and num == 0:
        with open('{}/graph.dot'.format(args.result_dir), 'w') as o:
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

  if train and (epoch == 1 or epoch % args.snapshot == 0):
    model_fn = '{}/epoch-{}.model'.format(args.result_dir, epoch)
    opt_fn = '{}/epoch-{}.state'.format(args.result_dir, epoch)
    serializers.save_hdf5(model_fn, model)
    serializers.save_hdf5(opt_fn, optimizer)

  if train:
    logging.info('epoch:{}\ttrain loss:{:.4f}\ttrain accuracy:{:.4f}'.format(
      epoch, sum_loss / num, sum_accuracy / num))
  else:
    logging.info('epoch:%d\ttest accuracy:%0.4f'%(epoch,sum_accuracy/num))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='models/VGG.py')
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--epoch', type=int, default=170)
  parser.add_argument('--batchsize', type=int, default=128) #128
  parser.add_argument('--snapshot', type=int, default=10)
  parser.add_argument('--datadir', type=str, default='data')
  parser.add_argument('--augment', action='store_true', default=False)

  # optimization
  parser.add_argument('--opt', type=str, default='MomentumSGD',
            choices=['MomentumSGD', 'Adam', 'AdaGrad'])
  parser.add_argument('--weight_decay', type=float, default=0.0001)
  parser.add_argument('--alpha', type=float, default=0.001)
  parser.add_argument('--lr', type=float, default=0.1)
  parser.add_argument('--lr_decay_freq', type=int, default=80)
  parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
  parser.add_argument('--validate_freq', type=int, default=1)
  parser.add_argument('--seed', type=int, default=1701)

  args = parser.parse_args()
  np.random.seed(args.seed)
  # os.environ['CHAINER_TYPE_CHECK'] = str(args.type_check)
  # os.environ['CHAINER_SEED'] = str(args.seed)

  # create result dir
  create_result_dir(args)

  # create model and optimizer
  model, optimizer = get_model_optimizer(args)

  dataset = load_hdf5(args.augment)
  tr_data, tr_labels, te_data, te_labels = dataset

  # learning loop
  for epoch in range(1, args.epoch + 1):
    logging.info('learning rate:{}'.format(optimizer.lr))

    one_epoch_resnet(args,model,optimizer,tr_data,tr_labels,epoch,True)
    model.save()
    # one_epoch(args, model, optimizer, tr_data, tr_labels, epoch, True)

    if epoch == 1 or epoch % args.validate_freq == 0:
      one_epoch_resnet(args, model, optimizer, te_data, te_labels, epoch, False)

    if args.opt == 'MomentumSGD' and epoch % args.lr_decay_freq == 0:
      optimizer.lr *= args.lr_decay_ratio
