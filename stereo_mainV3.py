from os.path import isfile, join
import numpy as np
import cv2 as cv
from cv2 import ximgproc
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
VIDEO_SIZE = (640, 480)
OUTPUT_PATH = 'path_planning_result_fast.avi'

def rescaleROI(src, roi):
    x, y, w, h = roi
    return src[y:y+h, x:x+w]

def main():
    streamFrames = range(0, 98)

    if SAVE_VIDEO:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(OUTPUT_PATH, fourcc, FPS, VIDEO_SIZE)

    for frameId in streamFrames:
        fnameL = str(frameId) + '.jpg'
        fnameR = str(frameId + 1) + '.jpg'
        if not (isfile(join(PATH_L, fnameL)) and isfile(join(PATH_R, fnameR))):
            print(f'Skipping frame {frameId}, image not found.')
            continue

        frame = compute_disparity(fnameL, params)

        if SAVE_VIDEO:
            out.write(frame)

    if SAVE_VIDEO:
        out.release()
        print(f'[INFO] Fast video saved to: {OUTPUT_PATH}')

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

    dispL = stereoL.compute(grayL, grayR)
    dispR = stereoR.compute(grayR, grayL)
    dispFinal = wls.filter(dispL, imgL, None, dispR)

    points3d = cv.reprojectImageTo3D(dispFinal, Q, ddepth=cv.CV_32F, handleMissingValues=True)
    frame = draw_fast_path(imgL, points3d, nDisp)
    return frame

def draw_fast_path(img, points3d, nDisp):
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
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, _ = finder.find_path(start, end, mat_grid)

    if len(path) > 1:
        for i in range(1, len(path)):
            pt1 = (path[i-1].x, VIDEO_SIZE[1] - path[i-1].y)
            pt2 = (path[i].x, VIDEO_SIZE[1] - path[i].y)
            cv.line(img, pt1, pt2, (0, 0, 255), 2)

    cv.putText(img, f'Path points: {len(path)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return cv.resize(img, VIDEO_SIZE)

main()
