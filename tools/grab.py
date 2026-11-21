"""GrabCut inside a loose box: python grab.py image.jpg left top right bottom out.jpg"""
import sys
import cv2
import numpy as np
img = cv2.imread(sys.argv[1]); h, w = img.shape[:2]
l, t, r, b = map(float, sys.argv[2:6])
m = 0.15
x0, y0 = int(max(0, (l - m * (r - l)) * w)), int(max(0, (t - m * (b - t)) * h))
x1, y1 = int(min(w - 1, (r + m * (r - l)) * w)), int(min(h - 1, (b + m * (b - t)) * h))
mask = np.zeros((h, w), np.uint8)
bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
cv2.grabCut(img, mask, (x0, y0, x1 - x0, y1 - y0), bg, fg, 5, cv2.GC_INIT_WITH_RECT)
obj = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
n, lab, stats, _ = cv2.connectedComponentsWithStats(obj)
k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
X, Y, W, H = stats[k, :4]
print({"left": round(X / w, 3), "top": round(Y / h, 3), "right": round((X + W) / w, 3), "bottom": round((Y + H) / h, 3)})
out = img.copy(); out[lab == k] = (0.5 * out[lab == k] + [0, 127, 0]).astype(np.uint8)
cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
cv2.imwrite(sys.argv[6], out)
