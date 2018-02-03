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


class DataLoader(object):

    def __init__(self):
        if osp.isfile('cache/shp.pkl'):
            with open('cache/shp.pkl', 'r') as fp:
                self.shp = cPickle.load(fp)
        else:
            self._load_shp()
            with open('cache/shp.pkl', 'w+') as fp:
                cPickle.dump(self.shp, fp, 2)
        if osp.isfile('cache/rgb.pkl'):
            with open('cache/rgb.pkl', 'r') as fp:
                self.rgb = cPickle.load(fp)
        else:
            self._load_rgb()
            with open('cache/rgb.pkl', 'w+') as fp:
                cPickle.dump(self.rgb, fp, 2)
        if osp.isfile('cache/chm.pkl'):
            with open('cache/chm.pkl', 'r') as fp:
                self.chm = cPickle.load(fp)
        else:
            self._load_chm()
            with open('cache/chm.pkl', 'w+') as fp:
                cPickle.dump(self.chm, fp, 2)
        if osp.isfile('cache/chm2.pkl'):
            with open('cache/chm2.pkl', 'r') as fp:
                self.chm2 = cPickle.load(fp)
        else:
            self._generate_chm()
            with open('cache/chm2.pkl', 'w+') as fp:
                cPickle.dump(self.chm2, fp, 2)

    def _load_shp(self):
        # Load shape annotations: {'001': [[(x,y)],]}
        # shapefile:
        # https://github.com/GeospatialPython/pyshp/blob/master/shapefile.py
        path = osp.join(config.datapath, "Task1/ITC")
        assert osp.isdir(path), path
        self.shp = {}
        for fn in glob(osp.join(path, "*.shp")):
            num = fn.split('_')[-1].split('.')[0]
            recs = shapefile.Reader(fn).shapeRecords()
            self.shp[num] = [r.shape.points for r in recs]

    def add_polygons_to_img(self, im, xy, polygons):
        im = im.copy()
        for p in polygons:
            xy_poly = []
            for x, y in p:
                x = int((x - xy[0]) / 0.25)
                y = int((xy[1] - y) / 0.25)
                xy_poly.append((x, y))
            for k in range(1, len(xy_poly)):
                cv2.line(im, xy_poly[k - 1], xy_poly[k], (0, 0, 255), 1)
        return im

    def _load_rgb(self):
        # Load rgb images: self.rgb = {'001':(tif, (xloc, yloc))}
        self.rgb = {}
        rgbpath = osp.join(config.datapath, 'RSdata/camera')
        assert osp.isdir(rgbpath), rgbpath
        for fn in glob(osp.join(rgbpath, "*.tif")):
            fp = gdal.Open(fn)
            xloc, _, _, yloc, _, _ = fp.GetGeoTransform()
            b = fp.GetRasterBand(1).ReadAsArray(
                0, 0, fp.RasterXSize, fp.RasterYSize).astype(np.uint8)
            g = fp.GetRasterBand(2).ReadAsArray(
                0, 0, fp.RasterXSize, fp.RasterYSize).astype(np.uint8)
            r = fp.GetRasterBand(3).ReadAsArray(
                0, 0, fp.RasterXSize, fp.RasterYSize).astype(np.uint8)
            img = cv2.merge([r, g, b])
            num = fn.split('_')[-2]
            self.rgb[num] = (img, (xloc, yloc))
            if num in self.shp and False:
                # Debug
                img = self.add_polygons_to_img(
                    img, (xloc, yloc), self.shp[num])
                cv2.imwrite('%srgb.jpg' % num, img)


    def _generate_chm(self):
        # Here we will generate chm our self to obtain higher resolutions.
        # Rules:
        #   (1) If it has rbg, then use the same resolution with rgb.
        #   (2) Else if its area is within 80x80, then use that area.
        #   (3) Else, use 80x80.
        # x is off by 10
        self.chm2 = {}  # {'001':(height_map, (xloc, yloc))}
        from sklearn.neighbors import KDTree
        pcpath = osp.join(config.datapath, 'RSdata/pointCloud')
        assert osp.isdir(pcpath), pcpath
        for fn in glob(osp.join(pcpath, "*.csv")):
            # for each ponint cloud.
            num = fn.split('_')[-1].split('.')[0]
            with open(fn, 'r') as f:
                lines = f.read().strip('\n').split('\n')[1:]
                XY = []
                Z = []
                for l in lines:
                    x, y, z = l.split(',')
                    XY.append((float(x), float(y)))
                    Z.append(float(z))
            if num in self.rgb:
                x0, y0 = self.rgb[num][1]
                numy, numx = self.rgb[num][0].shape[:2]
            else:
                x0 = min([x for x, y in XY])  # west most
                y0 = max([y for x, y in XY])  # north most
                numx = 320
                numy = 320
            x_target = [x0 - 1 + 0.25 * dx for dx in xrange(0, numx)]
            y_target = [y0 - 0.25 * dy for dy in xrange(0, numy)]
            xy_target = [(x, y) for y in y_target for x in x_target]
            kdt = KDTree(XY, leaf_size=30, metric='euclidean')
            indices = kdt.query(xy_target, k=1, return_distance=False)
            Z = np.array(Z, dtype=np.float)
            heights = np.array(Z[indices], dtype=np.float)
            min_heights = sorted(heights, reverse=False)
            th = int(0.01 * len(min_heights))
            assert th > 1
            min_heights = np.mean(min_heights[:th])
            heights -= min_heights
            heights = [h if h >= 0.0 else 0.0 for h in heights]
            heights = np.array(heights, dtype=np.float)
            heights = heights.reshape((numy, numx))
            self.chm2[num] = (heights, (x0, y0))


    def _load_chm(self):
        # Load chm data (containing z height): self.chm = {'001':z}
        self.chm = {}
        chmpath = osp.join(config.datapath, 'RSdata/chm')
        assert osp.isdir(chmpath), rgbpath
        for fn in glob(osp.join(chmpath, "*.tif")):
            fp = gdal.Open(fn)
            xloc, _, _, yloc, _, _ = fp.GetGeoTransform()
            z = fp.GetRasterBand(1).ReadAsArray(
                0, 0, fp.RasterXSize, fp.RasterYSize).astype(np.uint8)
            num = fn.split('_')[-2]
            self.chm[num] = (z, (xloc, yloc))
