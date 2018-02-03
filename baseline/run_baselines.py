from DataLoader import DataLoader
import sys
import time
import os
import gdal
from termcolor import colored
from config import config
from os import path as osp
from glob import glob
import cv2
import numpy as np
from easydict import EasyDict as edict
import cPickle
import shapefile
from shapely.geometry import Polygon
import copy
from sklearn.metrics.pairwise import euclidean_distances
from shapely.geometry import Polygon


def _cvt2img(mat):
    cv2.imwrite("/tmp/img.jpg", mat)
    return cv2.imread("/tmp/img.jpg")


def seg_chan_vese(chm_list):
    # chm_list: {'001': [[(x,y)],]}
    chan_vese_poly = {}  # {'001': [[(x,y)],]}
    for num, val in chm_list.items():
        heights = val[0]
        x0, y0 = val[1]
        imgarr = (heights - np.min(heights))
        imgarr = imgarr / np.max(imgarr) * 255
        imgarr = 255 - imgarr
        # segmentation
        from skimage.segmentation import chan_vese
        cv = chan_vese(imgarr, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
                       dt=0.5, init_level_set="checkerboard", extended_output=True)
        im = np.array(cv[0], dtype=np.int) * 255
        # create plogyon
        im = _cvt2img(im)
        im2 = cv2.copyMakeBorder(
            im, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(im2, 30, 200)
        (cnts, _) = cv2.findContours(
            edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = [c for c in cnts if cv2.contourArea(
            c) >= config.task1.minContourArea and
                cv2.contourArea(c) < config.task1.maxContourArea]
        polygons = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            pnts = cv2.approxPolyDP(c, 0.01 * peri, True)
            # pnts = cv2.convexHull(c)
            tmp = []
            for p in pnts:
                x = x0 + 0.25*(p[0][0] - 10) # -1 misalign, 0.25m/pixel, 10 border.
                y = y0 - 0.25*(p[0][1] - 10)
                tmp.append((x,y))
            pt = Polygon(tmp)
            if pt.is_valid == True:
                polygons.append(np.array(tmp))
        # vis = dl.add_polygons_to_img(im, (x0,y0), polygons)
        # cv2.imwrite('sample.jpg', vis)
        # exit(0)
        chan_vese_poly[num] = polygons
    return chan_vese_poly


def run_task1(dl):
    t0 = time.time()
    print "[Info] Running task1 ...",
    sys.stdout.flush()
    polygons = seg_chan_vese(copy.deepcopy(dl.chm2))
    w = shapefile.Writer(shapeType=5)
    w.field("plotnumber", "C")
    for i, ps in polygons.items():
        for p in ps:
            p = np.array(p)
            w.poly(parts=p.reshape(len(p), 1, 2).tolist())
            w.record("%s" % i)
        w.save(osp.join(config.outputpath, 'shp_%s.shp' % i))
    print "Done (%d images). Elapsed time is %.2fs" %\
        (len(polygons), time.time() - t0)


def run_task2():
    # approach description: use the nearest neighbor alignment.
    mode = 'test'
    t_start = time.time()
    print "[Info] Running task2 ...",
    sys.stdout.flush()
    pwd = osp.join(config.datapath, 'Task2/GroundData')
    # Get name and coordinate of stems.
    stem_names = []
    stem_xy = []
    stemfile = osp.join(pwd, 'ground_%s.csv' % mode)
    with open(stemfile, 'r') as fp:
        lines = fp.read().strip().split('\n')[1:]
        for l in lines:
            stemid, E, N = l.split(',')[:3]
            stem_names.append(stemid.strip())
            stem_xy.append(np.array((float(E.strip()), float(N.strip()))))
    no_of_stems = len(stem_names)
    # Get name and coordiate of crowns.
    crown_names = []
    crown_xy = []
    crownfile = osp.join(pwd, 'itc_attributes_%s.csv' % mode)
    with open(crownfile, 'r') as fp:
        lines = fp.read().strip().split('\n')[1:]
        for l in lines:
            plotid, crownid, E, N = l.split(',')[:4]
            crown_names.append(crownid.strip())
            crown_xy.append(np.array((float(E.strip()), float(N.strip()))))
    no_of_crowns = len(crown_names)
    # Sanity checks.
    if len(set(stem_names)) != len(stem_names):
        print colored("[Warn] There are duplicated stems in %s" % stemfile,
                      "yellow")
    if len(set(crown_names)) != len(crown_names):
        print colored("[Warn] There are duplicated crowns in %s" % crownfile,
                      "yellow")
    # A baseline alignment: by euclidean distance. stem(X)->crown(Y)
    alignment_matrix = np.empty(
        (no_of_stems + 1, no_of_crowns + 1), dtype=object)
    alignment_matrix[0] = ['ALIGN'] + crown_names
    e_dists = euclidean_distances(stem_xy, crown_xy)
    closest_crown_indices = np.argmin(e_dists, axis=1)
    for i_stem, xy in enumerate(stem_xy):
        alignment_matrix[i_stem + 1][0] = stem_names[i_stem]
        alignment_matrix[i_stem + 1][1:] = [0.0] * no_of_crowns
        alignment_matrix[i_stem + 1][1 + closest_crown_indices[i_stem]] = 1.0
    outfile = osp.join(config.outputpath, 'alignment_out.csv')
    np.savetxt(outfile, alignment_matrix, fmt='%s', delimiter=',')
    print "Done. Elapsed time is %.2fs" % (time.time() - t_start)

def run_task3():
    # approach: prior of the classes.
    t_start = time.time()
    print "[Info] Running task3 ...",
    sys.stdout.flush()
    with open(osp.join(config.datapath,
                       'Task3/GroundData/species_id_train.csv')) as fp:
        lines = fp.read().strip('\n').split('\n')[1:]
        species = []
        species2id = {}
        for l in lines:
            c, s, g, sid, gid = l.split(',')
            species.append(s)
            species2id[s] = sid
        num_train_points = len(lines)
        from collections import Counter
        species_cnt = Counter(species)
        species_prob = {k: float(species_cnt[k])/num_train_points for k in
                        species_cnt.keys()}
        species_list = species_prob.keys()
        species_prior = [species_prob[k] for k in species_list]
        species_id_list = [species2id[s] for s in species_list]
    # Load test data
    with open(osp.join(config.datapath,
                       'Task3/GroundData/hyper_bands_test.csv')) as fp:
        lines = fp.read().strip('\n').split('\n')[1:]
        crownids = list(set([l.split(',')[0].strip() for l in lines]))
        num_test_points = len(crownids)

    # Save output: stem->species_id cvs matrix
    num_species = len(species_cnt)
    cls_matrix = np.empty(
        (num_test_points + 1, num_species + 1), dtype=object)
    cls_matrix[0] = ['CLS'] + species_id_list
    for k in xrange(1, 1+num_test_points):
        cls_matrix[k][0] = crownids[k-1]
        cls_matrix[k][1:] = species_prior
    outfile = osp.join(config.outputpath, 'classification_out.csv')
    np.savetxt(outfile, cls_matrix, fmt='%s', delimiter=',')
    print "Done. Elapsed time is %.2fs" % (time.time() - t_start)


if __name__ == "__main__":
    os.system("mkdir -p %s 2>/dev/null" % config.outputpath)
    # Load data
    t0 = time.time()
    print "[Info] Loading data ...",
    sys.stdout.flush()
    dl = DataLoader()
    print "Done. Elapsed time is %.2fs" % (time.time() - t0)
    # Task #1
    run_task1(dl)
    # Task #2
    run_task2()
    # Task #3
    run_task3()
