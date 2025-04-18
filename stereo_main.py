from os.path import isfile, join
import numpy as np
import cv2 as cv
from cv2 import ximgproc
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import time

# Global Variables
PATH_L = r'C:\\Users\\ShubhamTaneja\\backendModels\\L_stream'
PATH_R = r'C:\\Users\\ShubhamTaneja\\backendModels\\R_stream'
PATH_CALIB = r'./Calibration_Files_expm'

# Stereo Matcher Parameters
minDisp = 0
nDisp = 96
bSize = 9
P1 = 8 * 3 * bSize ** 2
P2 = 32 * 3 * bSize ** 2
modeSgbm = cv.StereoSGBM_MODE_SGBM
pfCap = 0
sRange = 0
yfloor = 340

# WLS Filter Parameters
lam = 32000
sigma = 2.5
discontinuityRad = 4

params = [minDisp, nDisp, bSize, pfCap, sRange]

# Load Calibration
undistL = np.loadtxt(join(PATH_CALIB, 'umapL.txt'), dtype=np.float32)
rectifL = np.loadtxt(join(PATH_CALIB, 'rmapL.txt'), dtype=np.float32)
undistR = np.loadtxt(join(PATH_CALIB, 'umapR.txt'), dtype=np.float32)
rectifR = np.loadtxt(join(PATH_CALIB, 'rmapR.txt'), dtype=np.float32)
roiL = np.loadtxt(join(PATH_CALIB, 'ROIL.txt'), dtype=np.int64)
roiR = np.loadtxt(join(PATH_CALIB, 'ROIR.txt'), dtype=np.int64)
Q = np.loadtxt(join(PATH_CALIB, 'Q.txt'), dtype=np.float32)
RL = np.loadtxt(join(PATH_CALIB, 'RectifL.txt'), dtype=np.float32)
CL = np.loadtxt(join(PATH_CALIB, 'CmL.txt'), dtype=np.float32)
DL = np.loadtxt(join(PATH_CALIB, 'DcL.txt'), dtype=np.float32)

# Video Output Settings
SAVE_VIDEO = True
FPS = 10
VIDEO_SIZE = (1280, 720)
OUTPUT_PATH = 'path_planning_result2.avi'

def rescaleROI(src, roi):
    x, y, w, h = roi
    return src[y:y+h, x:x+w]

def main():
    streamFrames = range(0, 98)

    if SAVE_VIDEO:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(OUTPUT_PATH, fourcc, FPS, VIDEO_SIZE)

    plt.figure(figsize=(16, 9))

    for frameId in streamFrames:
        fnameL = str(frameId) + '.jpg'
        fnameR = str(frameId + 1) + '.jpg'
        if not (isfile(join(PATH_L, fnameL)) and isfile(join(PATH_R, fnameR))):
            print(f'Skipping frame {frameId}, image not found.')
            continue

        compute_disparity(fnameL, params)

        plt.draw()
        fig = plt.gcf()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img = cv.resize(img, VIDEO_SIZE)

        if SAVE_VIDEO:
            out.write(img)

        plt.clf()

    if SAVE_VIDEO:
        out.release()
        print(f'[INFO] Video saved to: {OUTPUT_PATH}')
    else:
        plt.show()

def compute_disparity(imgId, params):
    imgL = cv.imread(join(PATH_L, imgId))
    imgIdR = int(imgId.split('.')[0]) + 1
    imgIdR = str(imgIdR) + '.jpg'
    imgR = cv.imread(join(PATH_R, imgIdR))

    imgL = cv.remap(imgL, undistL, rectifL, cv.INTER_LINEAR)
    imgR = cv.remap(imgR, undistR, rectifR, cv.INTER_LINEAR)

    imgL = rescaleROI(imgL, roiL)
    imgR = rescaleROI(imgR, roiR)

    if imgL.shape != imgR.shape:
        dsize = (imgL.shape[1], imgL.shape[0])
        imgR = cv.resize(imgR, dsize, interpolation=cv.INTER_LINEAR)

    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    (minDisp, nDisp, bSize, pfCap, sRange) = params

    stereoL = cv.StereoSGBM_create(
        minDisparity=minDisp,
        numDisparities=nDisp,
        blockSize=bSize,
        P1=P1,
        P2=P2,
        speckleRange=sRange,
        preFilterCap=pfCap,
        mode=modeSgbm
    )

    wls = ximgproc.createDisparityWLSFilter(stereoL)
    stereoR = ximgproc.createRightMatcher(stereoL)
    wls.setLambda(lam)
    wls.setDepthDiscontinuityRadius(discontinuityRad)
    wls.setSigmaColor(sigma)

    ts1 = time.time()
    dispL = stereoL.compute(grayL, grayR)
    dispR = stereoR.compute(grayR, grayL)
    ts2 = time.time()
    cost_sgbm = ts2 - ts1

    dispFinal = wls.filter(dispL, imgL, None, dispR)
    dispFinal = ximgproc.getDisparityVis(dispFinal)

    points3d = cv.reprojectImageTo3D(dispFinal, Q, ddepth=cv.CV_32F, handleMissingValues=True)

    find_path(imgId, nDisp, points3d, dispFinal, cost_sgbm)

def find_path(imgId, nDisp, points3d, disparityMap, cost_sgbm):
    np.set_printoptions(suppress=True, precision=3)
    xx, yy, zz = points3d[:,:,0], points3d[:,:,1], points3d[:,:,2]
    xx, yy, zz = np.clip(xx, -25, 60), np.clip(yy, -25, 25), np.clip(zz, 0, 100)

    obs = zz[yfloor-10:yfloor,:]
    obstacles = np.amin(obs, 0, keepdims=False)
    y = np.mgrid[0:np.amax(obstacles), 0:obs.shape[1]][0,:,:]

    occupancy_grid = np.where(y >= obstacles, 0, 1)
    occupancy_grid[:, :nDisp+60] = 0

    far_zy, far_zx = np.unravel_index(np.argmax(np.flip(occupancy_grid[:,:-90])), occupancy_grid[:,:-90].shape)
    far_zx = (zz.shape[1]-91) - far_zx
    far_zy = occupancy_grid.shape[0] - far_zy - 1

    xcenter = 305

    mat_grid = Grid(matrix=occupancy_grid)
    start = mat_grid.node(xcenter, 1)
    end = mat_grid.node(far_zx, far_zy)
    tp1 = time.time()
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start, end, mat_grid)
    tp2 = time.time()
    cost_path = tp2-tp1

    if len(path) == 0:
        print('ERROR: No path found')

    coords = np.array([(xp, zp) for xp, zp in path], dtype=np.int32)
    yrange = np.geomspace(yy.shape[0]-1, yfloor+1, num=len(path), dtype=np.int32)
    yrange = np.flip(yrange)
    yworld = np.geomspace(10,13, num=len(path), dtype=np.float32)
    xworld = xx[yrange, coords[:,0]]
    zworld = np.array([zp for _, zp in path], dtype=np.float32)
    zworld = np.interp(zworld, [0, np.amax(zworld)], [25, nDisp])
    cf = np.array([xworld, yworld, zworld]).T

    pr, _ = cv.projectPoints(cf, np.zeros(3), np.zeros(3), CL, DL)
    pr = np.squeeze(pr, 1)
    py = pr[:,1]
    px = pr[:,0]

    fPts = np.array([[-40, 13, nDisp], [40, 13, nDisp], [40, 15, 0], [-40, 15, 0]], dtype=np.float32).T
    pf, _ = cv.projectPoints(fPts, np.zeros(3).T, np.zeros(3), CL, None)
    pf = np.squeeze(pf, 1)

    imL = cv.imread(join(PATH_L, imgId))
    imL = cv.cvtColor(imL, cv.COLOR_BGR2RGB)

    plt.clf()
    plt.suptitle(imgId.split('.')[0])
    costStats = '(far_zx, far_zy)=({},{})\ncost_path={:.3f}\ncost_sgbm={:.3f}'.format(far_zx, far_zy, cost_path, cost_sgbm)
    plt.gcf().text(x=0.6, y=0.05, s=costStats, fontsize='small')
    pathStats = 'steps={}\npathlen={}'.format(runs, len(path))
    plt.gcf().text(x=0.75, y=0.05, s=pathStats, fontsize='small')

    plt.subplot(221); plt.imshow(imL); plt.title('Planned Path (Left Camera)')
    plt.xlim([0, 1640]); plt.ylim([1232, 0])
    plt.scatter(px, py, s=np.geomspace(70, 5, len(px)), c=cf[:,1], cmap=plt.cm.plasma_r, zorder=99)
    plt.gca().add_patch(Polygon(pf, fill=True, facecolor=(0,1,0,0.12), edgecolor=(0,1,0,0.35)))

    ax = plt.gcf().add_subplot(222, projection='3d')
    ax.azim = 90; ax.elev = 110; ax.set_box_aspect((4,3,3))
    ax.plot_surface(xx[100:yfloor,:], yy[100:yfloor,:], zz[100:yfloor,:], cmap=plt.cm.viridis_r, rcount=25, ccount=25, linewidth=0, antialiased=False)
    ax.set_xlabel('Azimuth (X)'); ax.set_ylabel('Elevation (Y)'); ax.set_zlabel('Depth (Z)')
    ax.invert_xaxis(); ax.invert_zaxis(); ax.set_title('Planned Path (wrt. world-frame)')
    ax.scatter3D(cf[:,0], cf[:,1], cf[:,2], c=cf[:,2], cmap=plt.cm.plasma_r)

    plt.subplot(223); plt.imshow(disparityMap); plt.title('WLS Filtered Disparity Map')

    plt.subplot(224); plt.imshow(occupancy_grid, origin='lower', interpolation='none')
    plt.title('Occupancy Grid with A* Path')
    plt.plot(coords[:,0], coords[:,1], 'r')

main()
