import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import json

# --------------------------
# Helper functions
# --------------------------

def load_answer_key_excel(file):
    df = pd.read_excel(file)
    subjects = df.columns.tolist()
    key_data = {"version_A": {"answers": []}}
    
    for i in range(len(df)):
        for subj in subjects:
            val = df.at[i, subj]
            try:
                ans = val.split('-')[1].strip().lower()
                if ',' in ans:
                    ans = ans.replace(' ','').split(',')
                key_data["version_A"]["answers"].append(ans)
            except Exception:
                key_data["version_A"]["answers"].append("")
    return key_data

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

def find_contours(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = [c for c in contours if 200 < cv2.contourArea(c) < 10000]
    return bubble_contours

def sort_contours(contours, method="top-to-bottom"):
    if len(contours) == 0:
        return []
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
                                             key=lambda b: b[1][i], reverse=reverse))
    return contours

def evaluate_omr(image, key_answers):
    thresh = preprocess_image(image)
    contours = find_contours(thresh)
    contours = sort_contours(contours, method="top-to-bottom")

    student_answers = []
    for c in contours:
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        total = cv2.countNonZero(mask & thresh)
        if total > 500:
            student_answers.append("marked")
        else:
            student_answers.append("")

    scores = []
    for idx, ans in enumerate(key_answers["version_A"]["answers"]):
        student_ans = student_answers[idx] if idx < len(student_answers) else ""
        if isinstance(ans, list):
            scores.append(1 if student_ans else 0)
        else:
            scores.append(1 if student_ans else 0)

    subject_scores = {}
    subjects = ["Python","EDA","SQL","Power BI","Statistics"]
    for i, subj in enumerate(subjects):
        subject_scores[subj] = sum(scores[i*20:(i+1)*20])

    total_score = sum(scores)
    return subject_scores, total_score

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Automated OMR Evaluation", layout="wide")
st.title("Automated OMR Evaluation & Scoring System")

st.sidebar.header("Upload Files")

key_file = st.sidebar.file_uploader("Answer Key (Excel)", type=["xlsx"], key="key_uploader")

# Option 1: Upload OMR sheet images
omr_files = st.sidebar.file_uploader("Upload OMR Sheets (Images)", type=["jpg","jpeg","png"], accept_multiple_files=True, key="omr_uploader")

# Option 2: Capture OMR sheet using camera
camera_photo = st.sidebar.camera_input("Take a Photo of OMR Sheet")

if key_file is not None:
    key_data = load_answer_key_excel(key_file)
    st.success("Answer key loaded successfully!")

    results = []

    # Process uploaded files
    if omr_files:
        for file in omr_files:
            image = np.array(Image.open(file).convert("RGB"))
            image = cv2.resize(image, (800, 1200))
            subj_scores, total = evaluate_omr(image, key_data)
            row = {"Student": file.name, **subj_scores, "Total": total}
            results.append(row)

    # Process camera photo
    if camera_photo:
        image = np.array(Image.open(camera_photo).convert("RGB"))
        image = cv2.resize(image, (800, 1200))
        subj_scores, total = evaluate_omr(image, key_data)
        row = {"Student": "Camera_Capture", **subj_scores, "Total": total}
        results.append(row)

    if results:
        df_results = pd.DataFrame(results)
        st.subheader("Evaluation Results")
        st.dataframe(df_results)

        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download CSV", data=csv, file_name="omr_results.csv", mime="text/csv")
