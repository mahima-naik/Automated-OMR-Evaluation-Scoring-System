# omr.py
import cv2
import numpy as np
from skimage.filters import threshold_local
import os

FIXED_WIDTH = 1200
FIXED_HEIGHT = 1700

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def auto_warp(image):
    """Detect largest 4-point contour and warp. Fallback: center-crop/resize."""
    orig = image.copy()
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        # fallback: resize and return
        warped_color = cv2.resize(orig, (FIXED_WIDTH, FIXED_HEIGHT))
        warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
        return warped_color, warped_gray

    pts = screenCnt.reshape(4,2).astype("float32")
    rect = order_points(pts)
    dst = np.array([[0,0],[FIXED_WIDTH-1,0],[FIXED_WIDTH-1,FIXED_HEIGHT-1],[0,FIXED_HEIGHT-1]],dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (FIXED_WIDTH, FIXED_HEIGHT))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return warped, warped_gray

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def threshold_image(gray):
    """Adaptive threshold with local gaussian & invert so filled marks are white."""
    T = threshold_local(gray, 35, offset=10, method='gaussian')
    th = (gray > T).astype('uint8') * 255
    th = cv2.bitwise_not(th)
    return th

def find_bubble_contours(thresh, min_area=300, max_area=5000):
    """Find contours likely to be bubbles (circular-ish)."""
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbleCnts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        aspect = w/float(h) if h>0 else 0
        if 0.6 <= aspect <= 1.4 and w>10 and h>10:
            bubbleCnts.append(c)
    return bubbleCnts

def sort_contours(cnts, method="top-to-bottom"):
    boxes = [cv2.boundingRect(c) for c in cnts]
    if method=="top-to-bottom":
        cnts, boxes = zip(*sorted(zip(cnts, boxes), key=lambda b: b[1][1]))
    elif method=="left-to-right":
        cnts, boxes = zip(*sorted(zip(cnts, boxes), key=lambda b: b[1][0]))
    return list(cnts), list(boxes)

def extract_answers(bubbleCnts, thresh_image, choices=4, questions=100):
    """
    From detected bubble contours and threshold image, return a list of length questions:
    values: 0..choices-1 (index of choice), or -1 for ambiguous/unfilled.
    Strategy: sort by y then x, chunk into groups of 'choices'.
    """
    if len(bubbleCnts) < choices:
        return [-1]*questions

    # sort top-to-bottom
    bubbleCnts, _ = sort_contours(bubbleCnts, method='top-to-bottom')
    # if total bubbles is not exactly choices*questions, we'll try best-effort by selecting largest N
    total_needed = choices * questions
    if len(bubbleCnts) < total_needed:
        # not enough bubbles found; continue with what we have and pad
        pass
    if len(bubbleCnts) > total_needed:
        # try to keep the most relevant by selecting contours with largest area (assume template bubbles)
        areas = [cv2.contourArea(c) for c in bubbleCnts]
        idx_sorted = np.argsort(areas)[::-1][:total_needed]
        bubbleCnts = [bubbleCnts[i] for i in sorted(idx_sorted)]
        # re-sort top-to-bottom
        bubbleCnts, _ = sort_contours(bubbleCnts, method='top-to-bottom')

    answers = []
    idx = 0
    for q in range(questions):
        group = bubbleCnts[idx:idx+choices]
        if len(group) < choices:
            answers.append(-1)
            idx += choices
            continue
        idx += choices
        # sort left-to-right within group
        g_sorted, boxes = sort_contours(group, method='left-to-right')
        filled_values = []
        for c in g_sorted:
            mask = np.zeros(thresh_image.shape, dtype='uint8')
            cv2.drawContours(mask, [c], -1, 255, -1)
            # count non-zero (white) pixels in threshold image (filled marks are white after invert)
            filled = cv2.countNonZero(cv2.bitwise_and(thresh_image, thresh_image, mask=mask))
            filled_values.append(filled)
        filled_values = np.array(filled_values)
        max_idx = int(np.argmax(filled_values))
        max_val = int(filled_values[max_idx])
        second = int(np.sort(filled_values)[-2]) if len(filled_values) > 1 else 0

        # heuristics: threshold on absolute count & gap to second best to reduce errors
        if max_val < 200 or (max_val - second) < 80:
            answers.append(-1)
        else:
            answers.append(max_idx)
    return answers

def draw_overlay(warped_color, bubbleCnts, answers, letters=['A','B','C','D']):
    """Draw bounding boxes around bubbles and mark detected choice per question. Return overlay image."""
    overlay = warped_color.copy()
    # annotate all bubbles
    try:
        bubbleCnts_sorted, _ = sort_contours(bubbleCnts, method='top-to-bottom')
    except Exception:
        bubbleCnts_sorted = bubbleCnts
    idx = 0
    for q, ans in enumerate(answers):
        for i in range(4):
            if idx >= len(bubbleCnts_sorted):
                break
            c = bubbleCnts_sorted[idx]
            x,y,w,h = cv2.boundingRect(c)
            color = (0,255,0)
            thickness = 1
            cv2.rectangle(overlay, (x,y), (x+w,y+h), color, thickness)
            if ans == i:
                cv2.circle(overlay, (x + w//2, y + h//2), max(3, min(w,h)//3), (0,0,255), 2)
                cv2.putText(overlay, letters[i], (x,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
            idx += 1
    return overlay

# utility to create directories
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
