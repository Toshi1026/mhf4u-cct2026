import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
import core
from cutB.beats import _useA  # noqa
from cutB.beats import _figures as F, _render as R
bg = np.zeros((700, 1080, 3), np.float32) + core.c('5A9FD6')
def put(f, rgba, x, y):
    h, w = rgba.shape[:2]; a = rgba[..., 3:4]
    f[y:y+h, x:x+w] = f[y:y+h, x:x+w]*(1-a) + rgba[..., :3]*a
t=time.time(); v, x0, y0 = F.villain_relief(150, 300, 12.0, 0, 'stand', 0.3); print('villain stand', time.time()-t, v.shape)
put(bg, v, 0, 0)
t=time.time(); v2, _, _ = F.villain_relief(150, 300, 10.0, 30, 'fly', 0.3, cape_ang=200); print('villain fly', time.time()-t)
put(bg, v2, 330, 0)
t=time.time(); fi, ax, ay = F.fist_relief(-27, 1.6); print('fist', time.time()-t, fi.shape)
put(bg, fi, 640, 100)
t=time.time(); el, ax, ay = F.elephant_relief(2.4); print('ele', time.time()-t, el.shape)
put(bg, el, 300, 430)
t=time.time(); el, ax, ay = F.elephant_relief(0.82); print('ele small', time.time()-t, el.shape)
put(bg, el, 800, 600)
core.save_png(bg, '/home/user/mhf4u-cct2026/exports/v3/lookdev_B/_t_fig.png')
