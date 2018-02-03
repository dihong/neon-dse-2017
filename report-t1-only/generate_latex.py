import os
import sys
from glob import glob
from termcolor import colored
import gdal
from os import path as osp
import cv2
import numpy as np
from easydict import EasyDict as edict
import cPickle
import shapefile
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment
import operator
import matplotlib
from matplotlib import pyplot as plt
import pylab as py


def evaluate_jaccard(P1, P2, name):
    # P1 and P2 are groundtruth and prediction polygons for task 1.
    # Return: optimal Jaccard, TP, FP, FN in areas.
    # In case there is no corresponding prediction P2, it still works.
    N1 = len(P1)
    N2 = len(P2)
    if N1 > N2:
        cost_mat = np.zeros((N1, N1), dtype=np.float)
    else:
        cost_mat = np.zeros((N2, N2), dtype=np.float)
    # eval: cost_mat[i,j] is the negative J between P1[i] and P2[j]
    # TP, FP, FN
    TP = np.zeros(cost_mat.shape)
    FP = np.zeros(cost_mat.shape)
    FN = np.zeros(cost_mat.shape)
    for i, p1 in enumerate(P1):
        p1 = Polygon(p1)
        assert p1.is_valid
        a1 = p1.area
        for j, p2 in enumerate(P2):
            if p2.is_valid == False:
                print colored("Skipping invalid polygon in %s." % name, 'yellow')
                continue
            # assert p2.is_valid, P2
            a2 = p2.area
            I = p1.intersection(p2).area
            U = p1.union(p2).area
            assert U >= I
            J = I / float(U)
            cost_mat[(i, j)] = -J
            TP[(i, j)] = I
            FP[(i, j)] = a2 - I
            FN[(i, j)] = a1 - I
    # run hungarian algorithm to find the optimal cost.
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    J = (-1) * cost_mat[row_ind, col_ind].sum() / float(N1)
    TP = TP[row_ind, col_ind].sum()
    FP = FP[row_ind, col_ind].sum()
    FN = FN[row_ind, col_ind].sum()
    return J, TP, FP, FN


def add_polygons_to_img(im, xy, polygons, color=(0, 0, 255)):
    # im: cv2 image
    # xy: base offset (40xxx, 32xxx)
    # polygons: [[(x,y)],] in absolute coordinate.
    for p in polygons:
        xy_poly = []
        for x, y in p:
            x = int((x - xy[0]) / 0.25)
            y = int((xy[1] - y) / 0.25)
            xy_poly.append((x, y))
        for k in range(1, len(xy_poly)):
            cv2.line(im, xy_poly[k - 1], xy_poly[k], color, 1)


def draw_polygons(fn, shp_gt, shp_pd, pid):
    fp = gdal.Open(fn)
    xloc, _, _, yloc, _, _ = fp.GetGeoTransform()
    b = fp.GetRasterBand(1).ReadAsArray(
        0, 0, fp.RasterXSize, fp.RasterYSize).astype(np.uint8)
    g = fp.GetRasterBand(2).ReadAsArray(
        0, 0, fp.RasterXSize, fp.RasterYSize).astype(np.uint8)
    r = fp.GetRasterBand(3).ReadAsArray(
        0, 0, fp.RasterXSize, fp.RasterYSize).astype(np.uint8)
    img = cv2.merge([r, g, b])
    # draw polygons
    add_polygons_to_img(img, (xloc, yloc), shp_gt[pid], color=(0, 255, 0))
    if pid in shp_pd:
        add_polygons_to_img(img, (xloc, yloc), shp_pd[pid])
    return img


def task1(config):
    outpath = config.outputpath
    # The gt folder should only contain testing sample shape files.
    gtpath = osp.join(config.gtpath, 'Task1/ITC')
    assert osp.isdir(gtpath), gtpath
    spfiles = glob(osp.join(gtpath, '*.shp'))
    assert len(spfiles) > 0, gtpath
    shp_gt = {}
    for fn in spfiles:
        num = fn.split('_')[-1].split('.')[0]
        recs = shapefile.Reader(fn).shapeRecords()
        shp_gt[num] = [r.shape.points for r in recs]
    # Load prediction.
    pdpath = osp.join(config.income)
    assert osp.isdir(pdpath), pdpath
    spfiles = glob(osp.join(pdpath, '*.shp'))
    assert len(spfiles) > 0, pdpath
    shp_pd = {}
    for fn in spfiles:
        num = fn.split('_')[-1].split('.')[0]
        if num not in shp_gt:
            continue
        recs = shapefile.Reader(fn).shapeRecords()
        shp_pd[num] = [r.shape.points for r in recs]
    ss = (len(shp_gt), len(shp_pd))
    if len(shp_gt) != len(shp_pd):
        print colored('[Warn] #gt != #prediction: %d vs %d' % ss, 'yellow')
        for k in shp_gt.keys():
            if k not in shp_pd:
                print colored('Plot %03d does not have submissioin.' % int(k), 'yellow')
    else:
        print colored('[Info] Task1, loaded %d testing samples.' % ss[0],
                      'green')
    # For each gt plot, calculate the score with the corresponding pd plot.
    # If there is no corresponding pd plot, give warning.
    total_j = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    plotid_jaccard = []
    plotlevel_fpr = {}
    for num, P1 in shp_gt.items():
        if num in shp_pd:
            P2 = shp_pd[num]
        else:
            P2 = []
        # remove overlapping polygons.
        if len(P2) > 0:
            polys = [Polygon(p) for p in P2]
            sel = [0]
            for i in xrange(1, len(polys)):
                flag = True
                for j in sel:
                    if polys[i].intersects(polys[j]):
                        flag = False
                        break
                if flag:
                    sel.append(i)
            P2_unique = [polys[i] for i in sel]
        else:
            P2_unique = []
        #
        try:
            J, TP, FP, FN = evaluate_jaccard(P1, P2_unique, name=num)
        except:
            print colored("Error occured for plot num = %03d" % int(num), 'red')
            raise
        total_j += J
        total_tp += TP
        total_fp += FP
        total_fn += FN
        plotid_jaccard.append((num, J))
        plotlevel_fpr[num] = (TP, FP, FN)
    avg_j = total_j / len(shp_gt)
    # Save averaged jaccard, tp, fp and fn.
    with open(osp.join(outpath, "data/crown_delineation.dat"), "w+") as fp:
        fp.write("%.4f" % avg_j)
    # Save confusion matrix.
    table = []
    table.append("\\begin{tabularx}{0.9\\textwidth}{XXX}")
    table.append("\\textbf{Area (Square Meters)} & \\textbf{Positive} & \\textbf{Negative}"
                 + "\\\\")
    table.append("\\textbf{True} & %.1f & -\\\\" % total_tp)
    table.append("\\textbf{False} & %.1f & %.1f" % (total_fp, total_fn))
    table.append("\\end{tabularx}")
    with open(osp.join(outpath, "data/t1_conmat.dat"), "w+") as fp:
        fp.write("\n".join(table))
    # Save best 6 plots.
    rgbpath = osp.join(config.gtpath, 'Task1/RSdata/camera')
    plotid_jaccard_sorted = sorted(plotid_jaccard, key=operator.itemgetter(1),
                                   reverse=True)
    cntplot = 0
    for pid, jac in plotid_jaccard_sorted:
        fn = osp.join(rgbpath, 'OSBS_%s_camera.tif' % pid)
        if osp.isfile(fn) == False:
            continue
        img = draw_polygons(fn, shp_gt, shp_pd, pid)
        cv2.putText(img, "J = %.4f" % (jac),
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=2,
                    color=(255, 0, 0))
        cv2.putText(img, "%s" % pid,
                    tuple(reversed(np.array(img.shape[:2]) / 2 + [15, -60])),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255))
        cv2.imwrite(osp.join(outpath, 'figure', 'top5_%d.jpg' % cntplot), img)
        cntplot += 1
        if cntplot == 6:
            break
    # Save worest 6 plots.
    cntplot = 0
    for pid, jac in reversed(plotid_jaccard_sorted):
        fn = osp.join(rgbpath, 'OSBS_%s_camera.tif' % pid)
        if osp.isfile(fn) == False:
            continue
        img = draw_polygons(fn, shp_gt, shp_pd, pid)
        cv2.putText(img, "J = %.4f" % (jac),
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=2,
                    color=(255, 0, 0))
        cv2.putText(img, "%s" % pid,
                    tuple(reversed(np.array(img.shape[:2]) / 2 + [15, -60])),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255))
        cv2.imwrite(osp.join(outpath, 'figure',
                             'bottom5_%d.jpg' % cntplot), img)
        cntplot += 1
        if cntplot == 6:
            break
    # Save plot-level FPR
    plotlevel_fpr = sorted(plotlevel_fpr.items())
    xticks = [e[0] for e in plotlevel_fpr]
    plotLevelTruePositives = [e[1][0] for e in plotlevel_fpr]
    plotLevelFalsePositives = [e[1][1] for e in plotlevel_fpr]
    plotLevelFalseNegatives = [e[1][2] for e in plotlevel_fpr]
    num_plots = len(plotlevel_fpr)
    ind = np.arange(num_plots)
    assert len(ind) == len(xticks)
    width = 0.6
    p1 = plt.bar(ind, plotLevelTruePositives, width,
                 color='g', edgecolor="none")
    p2 = plt.bar(ind, plotLevelFalsePositives, width,
                 color='r', bottom=plotLevelTruePositives,
                 edgecolor="none")
    p3 = plt.bar(ind, plotLevelFalseNegatives, width,
                 color='b', bottom=np.array(
                     plotLevelTruePositives)
                 + np.array(plotLevelFalsePositives),
                 edgecolor="none")
    plt.xticks(ind + 0.25, xticks)
    plt.ylabel('Area in Square Meters')
    plt.xlabel('Plot ID')
    plt.title('Plot-Level Confusion Matrix')
    ymax = np.array(plotLevelTruePositives) + \
        np.array(plotLevelFalsePositives) + \
        np.array(plotLevelFalseNegatives)
    plt.ylim(1, 1.05 * max(ymax))
    plt.xlim(0, num_plots)
    plt.legend((p1[0], p2[0], p3[0]), ('TruePositive',
                                       'FalsePositive', 'FalseNegative'))
    fig = py.gcf()
    matplotlib.rcParams.update({'font.size': 16})
    fig.set_size_inches(18.5, 10.5, forward=True)
    py.savefig(osp.join(outpath, 'figure/confusionMatrix.svg'), format='svg',
               bbox_inches='tight')


def task2(config):
    outpath = config.outputpath
    # load the groundtruth mapping: stemid(fieldID) ->crownid (itcID)
    plotmap = {}  # {crownid:plotid}
    with open(osp.join(config.gtpath, 'Task2', 'GroundData',
                       'task2_test_results.csv')) as fp:
        lines = fp.read().strip().split('\n')[1:]
        stem2crown_gt = {l.split(',')[1]: l.split(',')[
            2].strip() for l in lines}
        plotmap = {l.split(',')[2].strip(): l.split(
            ',')[0].split('_')[-1] for l in lines}
    # Load predicted alignments
    A = np.loadtxt(osp.join(config.income, 'alignment_out.csv'),
                   dtype=object, delimiter=',')
    print colored("[Info] Task2, loaded alignment matrix of size %dx%d" %
                  A.shape, "green")
    # create axis-label to row/col index mappers.
    x_map = dict([(label, idx)
                  for idx, label in enumerate(A[0, 1:])])  # tree
    y_map = dict([(label, idx)
                  for idx, label in enumerate(A[1:, 0])])  # stem
    # normalize the A by row to sum up to 1.
    A_val = np.array(A[1:, 1:], dtype=float) + 0.000000001
    row_sums = A_val.sum(axis=1)
    A_norm = A_val / row_sums[:, np.newaxis]
    # calculate the scores.
    score = 0.0
    plotwise_accuracy = {}
    for stemid, y_idx in y_map.items():
        assert stemid in stem2crown_gt, stemid
        gt_crownid = stem2crown_gt[stemid]
        if gt_crownid not in x_map:
            print colored("[Error] crownid %s not found." % gt_crownid, 'red')
            exit(0)
        x_idx = x_map[gt_crownid]
        score += A_norm[(y_idx, x_idx)]
        plotid = plotmap[gt_crownid]
        if plotid not in plotwise_accuracy:
            plotwise_accuracy[plotid] = []
        plotwise_accuracy[plotid].append(A_norm[(y_idx, x_idx)])
    score /= len(y_map)
    items = plotwise_accuracy.items()
    for key, val in items:
        plotwise_accuracy[key] = np.mean(val)
    with open(osp.join(outpath, "data/crown_alignment.dat"), "w+") as fp:
        fp.write('%.4f' % score)
    # Plot.
    plotwise_accuracy = plotwise_accuracy.items()
    plotwise_accuracy = sorted(plotwise_accuracy)
    xticks = [k for k, v in plotwise_accuracy]
    yvals = [v for k, v in plotwise_accuracy]
    matplotlib.rcParams.update({'font.size': 14})
    plt.clf()
    plt.plot(range(len(plotwise_accuracy)),
             yvals, 'bd-',
             markersize=10, linewidth=1)
    ind = np.arange(len(xticks))
    plt.xticks(ind, xticks)
    ax = py.gca()
    ax.grid(color='k', linestyle='-.', linewidth=1)
    plt.xlabel("Plot ID")
    plt.ylabel("Average Alignment Scores")
    plt.title("Crown Alignment Performace by Plot ID\n")
    fig = py.gcf()
    fig.set_size_inches(16, 10, forward=True)
    py.savefig(osp.join(outpath, "figure/crown_alignment.svg"), format='svg',
               bbox_inches='tight')


def create_confusion_matrix_latex_table(mat, axis):
    N = mat.shape[1]
    assert N == len(axis)
    assert mat.shape[0] == mat.shape[1]
    table = []
    table.append("\\begin{tabularx}{\\textwidth}{r|%s}" % ''.join(['X'] * N))
    L = '{\\Large{Species ID}}&' + '&'.join(
        ["\\rotatebox[origin=l]{90}{%s}" % a.strip() for a in axis]) +\
        '\\\\\\hline'
    table.append(L)
    for i in xrange(N):
        tmp = []
        for j, v in enumerate(mat[i,:]):
            if j == i:
                tmp.append('\\textbf{%.2f}' % v)
            else:
                tmp.append('%.2f' % v)
        L = "%s&" % axis[i].strip() + '&'.join(tmp) + '\\\\'
        table.append(L)
    table.append("\\end{tabularx}")
    ret = "\n".join(table)
    return ret


def task3(config):
    outpath = config.outputpath
    # Load gt
    with open(osp.join(config.gtpath, 'Task3', 'GroundData',
                       'ResultsTestSet.csv'), 'r') as fp:
        lines = fp.read().strip().split('\r')
        gt_crownid2spid = {}
        for l in lines[1:]:
            _, crownid, _, speciesid, _ = l.split(',')
            gt_crownid2spid[crownid] = speciesid
        assert len(gt_crownid2spid) > 0
    # Load classifcation reseult
    cross_entropy = []
    rank_1 = 0
    with open(osp.join(config.income, 'species_id_subm.csv'), 'r') as fp:
        lines = fp.read().strip().split('\n')
        pd_spidlist = lines[0].strip().split(',')[1:]
        print colored("[Info] Task3, loaded classification matrix of size %dx%d"
                      % (len(lines) - 1, len(pd_spidlist)), "green")
        spidlist = sorted(pd_spidlist)
        spid2ind = {s: k for k, s in enumerate(spidlist)}
        num_sp = len(pd_spidlist)
        CM = np.zeros((num_sp, num_sp), dtype=np.float)
        num_test = len(lines) - 1
        for l in lines[1:]:
            elems = l.strip().split(',')
            crownid = elems[0]
            if crownid not in gt_crownid2spid:
                print colored('[Error] crownid in Task2 is invalid.' % crownid)
                exit(0)
            gt_spid = gt_crownid2spid[crownid]
            max_prob = -1
            max_spid = None
            for k, prob in enumerate(elems[1:]):
                if prob > max_prob:
                    max_prob = prob
                    max_spid = pd_spidlist[k]
                pd_spid = pd_spidlist[k]
                prob = float(prob)
                CM[spid2ind[pd_spid], spid2ind[gt_spid]] += prob
                if pd_spid == gt_spid:
                    cross_entropy.append(-np.log(0.001 + prob))
            if max_spid == gt_spid:
                rank_1 += 1
        assert len(cross_entropy) == num_test, "%d vs %d" %\
            (len(cross_entropy), num_test)
        cross_entropy = np.mean(cross_entropy)
        rank_1 /= float(num_test)
    # Evaluate metrics.
    eps = 1e-10
    # calculate per-class tp, tn, fp, fn.
    N = len(CM)
    CMS = np.sum(CM)
    TP = CM.diagonal()
    FP = np.sum(CM, axis=0) - TP
    FN = np.sum(CM, axis=1) - TP
    TN = CMS - TP - FP - FN
    # calculate per-class precision, recall, f1, accuracy, specificity, ck
    precision = TP / (eps + FP + TP)
    recall = TP / (eps + FN + TP)
    f1score = 2.0 * precision * recall / (precision + recall + eps)
    accuracy = (TP + TN) / float(CMS)
    specificity = FN / (eps + FN + FP)
    cohens_kappa = 1 - (1 - sum(TP) / float(CMS)) / \
        (1 - (CMS**2) / float(N**3))
    # save results.
    cm_latex = create_confusion_matrix_latex_table(CM, spidlist)
    with open(osp.join(outpath, "data/species_classification.dat"),
              "w+") as fp:
        fp.write("%.4f" % np.mean(cross_entropy))
    with open(osp.join(outpath, "data/t3_rank1.dat"), "w+") as fp:
        fp.write("%.4f" % np.mean(rank_1))
    with open(osp.join(outpath, "data/t3_accuracy.dat"), "w+") as fp:
        fp.write("%.4f" % np.mean(accuracy))
    with open(osp.join(outpath, "data/t3_cohens_kappa.dat"), "w+") as fp:
        fp.write("%.4f" % np.mean(cohens_kappa))
    with open(osp.join(outpath, "data/t3_f1.dat"), "w+") as fp:
        fp.write("%.4f" % np.mean(f1score))
    with open(osp.join(outpath, "data/t3_specificity.dat"), "w+") as fp:
        fp.write("%.4f" % np.mean(np.mean(specificity)))
    with open(osp.join(outpath, "data/confusion_matrix.dat"), "w+") as fp:
        fp.write(cm_latex)
    # plot precision, recall, f1
    plt.clf()
    width = 0.75
    xticks = [x.strip() + '--' for x in spidlist]
    ind = np.arange(N)
    p1 = plt.bar(ind - 0.38, f1score, width, color='g', edgecolor="none")
    plt.xticks(ind, xticks, rotation='vertical')
    plt.ylabel('F1 Score.')
    plt.xlabel('Species ID')
    ax = py.gca()
    ax.grid(color='k', linestyle='-.', linewidth=1)
    ymax = f1score
    plt.ylim(0, 1.05 * max(ymax))
    plt.xlim(-1, N)
    matplotlib.rcParams.update({'font.size': 20})
    fig = py.gcf()
    fig.set_size_inches(20, 10, forward=True)
    py.savefig(osp.join(outpath, 'figure/t3_f.svg'),
               format='svg', bbox_inches='tight')

    plt.clf()
    width = 0.75
    xticks = [x.strip() + '--' for x in spidlist]
    ind = np.arange(N)
    p1 = plt.bar(ind - 0.38, precision, width, color='b', edgecolor="none")
    plt.xticks(ind, xticks, rotation='vertical')
    plt.ylabel('Precision')
    plt.xlabel('Species ID')
    ax = py.gca()
    ax.grid(color='k', linestyle='-.', linewidth=1)
    ymax = precision
    plt.ylim(0, 1.05 * max(ymax))
    plt.xlim(-1, N)
    matplotlib.rcParams.update({'font.size': 20})
    fig = py.gcf()
    fig.set_size_inches(20, 10, forward=True)
    py.savefig(osp.join(outpath, 'figure/t3_p.svg'),
               format='svg', bbox_inches='tight')

    plt.clf()
    width = 0.75
    xticks = [x.strip() + '--' for x in spidlist]
    ind = np.arange(N)
    p1 = plt.bar(ind - 0.38, recall, width, color='r', edgecolor="none")
    plt.xticks(ind, xticks, rotation='vertical')
    plt.ylabel('Recall')
    plt.xlabel('Species ID')
    ax = py.gca()
    ax.grid(color='k', linestyle='-.', linewidth=1)
    ymax = recall
    plt.ylim(0, 1.05 * max(ymax))
    plt.xlim(-1, N)
    matplotlib.rcParams.update({'font.size': 20})
    fig = py.gcf()
    fig.set_size_inches(20, 10, forward=True)
    py.savefig(osp.join(outpath, 'figure/t3_r.svg'),
               format='svg', bbox_inches='tight')
    """
    plt.clf()
    width = 0.75
    xticks = [x.strip() + '--' for x in spidlist]
    ind = np.arange(N)
    p1 = plt.bar(ind - 0.38, f1score, width, color='g', edgecolor="none")
    p2 = plt.bar(ind - 0.38, precision, width, color='b',
                 bottom=f1score, edgecolor="none")
    p3 = plt.bar(ind - 0.38, recall, width, color='r',
                 bottom=precision + f1score, edgecolor="none")
    p4 = plt.plot(ind, f1score, 'gd--', markersize=15)
    plt.xticks(ind, xticks, rotation='vertical')
    plt.ylabel('Precision, Recall and F1 Scores.')
    plt.xlabel('Species ID')
    ax = py.gca()
    ax.grid(color='k', linestyle='-.', linewidth=1)
    ymax = precision + recall + f1score
    plt.ylim(0, 1.05 * max(ymax))
    plt.xlim(0, N)
    plt.legend((p1[0], p2[0], p3[0]), ('F1-Score', 'Precision', 'Recall'))
    matplotlib.rcParams.update({'font.size': 20})
    fig = py.gcf()
    fig.set_size_inches(20, 10, forward=True)
    py.savefig(osp.join(outpath, 'figure/t2_fpr.svg'),
               format='svg', bbox_inches='tight')
    """
    # plot specificity and accuracy
    plt.clf()
    p1 = plt.plot(ind, accuracy, "bs--", linewidth=1, markersize=10)
    p2 = plt.plot(ind, specificity, "r<--", linewidth=1, markersize=10)
    plt.xticks(ind, xticks, rotation='vertical')
    plt.ylim(0, 1.0)
    plt.xlim(0, N - 1)
    plt.legend((p1[0], p2[0]), ('Accuracy', 'Specificity'))
    plt.xlabel('Species ID')
    matplotlib.rcParams.update({'font.size': 16})
    ax = py.gca()
    ax.grid(color='k', linestyle='-.', linewidth=1)
    fig = py.gcf()
    fig.set_size_inches(20, 10, forward=True)
    py.savefig(osp.join(outpath, 'figure/t2_as.svg'),
               format='svg', bbox_inches='tight')


############################################################################

if __name__ == "__main__":
    # Select the tasks to run
    tasks = [1]
    # Read command line argument
    if len(sys.argv) != 2:
        print "Usage: python generate_latex.py name_of_team"
        exit(0)
    from config import config
    config.teamname = sys.argv[1]
    config.income = '../income/%s' % config.teamname
    config.outputpath = '../result/%s' % config.teamname
    # Check income
    if osp.isdir(config.income) == False:
        print colored("[Error] No submission was found for %s." % 
                      config.teamname, "red")
        exit(0)
    # Initialize output dir
    print colored("[Info] Generating report for %s ..." % config.teamname,
                  'green')
    outpath = config.outputpath
    os.system("mkdir -p %s 2>/dev/null" % outpath)
    os.system("cp -r latex-template/* %s" % outpath)
    os.system("cp -r ../default_output/* %s" % outpath)
    os.system("cp latex-template/dse_report.tex %s" % outpath)
    assert osp.isdir(config.gtpath), config.gtpath
    with open(osp.join(outpath, "data/teamname.dat"), "w+") as fp:
        fp.write(config.teamname)
    # Run tasks generator
    images = []
    if 1 in tasks:
        task1(config)
        images.append('confusionMatrix')
    if 2 in tasks:
        task2(config)
        images.append('crown_alignment')
    if 3 in tasks:
        task3(config)
        images.append('t3_f')
        images.append('t2_as')
        images.append('t3_p')
        images.append('t3_r')
    # Convert svg to pdf
    for i in images:
        os.system("inkscape -D -z --file=%s --export-pdf=%s --export-latex" %
                  (osp.join(outpath, "figure/%s.svg" % i),
                   osp.join(outpath, "%s.pdf" % i)))
    print colored("[Info] Running pdflatex...", "green")
    os.system("cd %s; pdflatex dse_report.tex >/dev/null" % outpath)
    print colored("[Info] PDF report saved to %s/dse_report.pdf" % outpath,
                  'green')
