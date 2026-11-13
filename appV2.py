import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import mediapipe as mp
import av
import math
import os
from datetime import datetime
import numpy as np
import plotly.graph_objects as go

# 1. ตั้งค่าหน้าเว็บแบบ Wide
st.set_page_config(page_title="Neck Align - ระบบวิเคราะห์มุมคอ 3D", page_icon="📐", layout="wide")

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0284c7 100%);
        padding: 24px 28px;
        border-radius: 18px;
        color: #ffffff !important;
        margin-bottom: 22px;
        box-shadow: 0 8px 25px rgba(2, 132, 199, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h2, .main-header p {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Session State สำหรับเก็บประวัติรูปภาพ และค่าเวกเตอร์ 3D ล่าสุด
if "captured_records" not in st.session_state:
    st.session_state.captured_records = []
if "active_vec" not in st.session_state:
    st.session_state.active_vec = [75, 120, -20]
    st.session_state.active_cva = 55.0

# โหลดโมเดล MediaPipe สำหรับภาพนิ่งไว้ล่วงหน้าแค่ครั้งเดียว (ป้องกันแอปค้าง)
@st.cache_resource
def get_static_pose_model():
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.4
    )

# 2. ส่วนหัวและตราโรงเรียน
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=110)
    else:
        st.markdown("<h1 style='text-align: center; font-size: 65px; margin: 0;'>🏫</h1>", unsafe_allow_html=True)
        st.caption("<center>*(ใส่ logo.png ในโฟลเดอร์)*</center>", unsafe_allow_html=True)

with col_title:
    st.markdown("""
        <div class="main-header">
            <h2 style='margin:0; font-weight:700; letter-spacing: 0.5px;'>📐 โครงงานคณิตศาสตร์: Neck Align</h2>
            <p style='margin:6px 0 0 0; font-size:15px; color:#e0f2fe;'>
                ระบบตรวจวัดและวิเคราะห์มุมกระดูกคอ (CVA) อัตโนมัติ พร้อม <b>การจำลองกระดูกสันหลังส่วนคอ C1–C7 ในปริภูมิ 3 มิติ</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

# 3. ฟังก์ชันคณิตศาสตร์และการจำลองกระดูกคอ C1–C7 ในรูปแบบ 3D
def calculate_cva_angle(ear_x, ear_y, sh_x, sh_y):
    dx = ear_x - sh_x
    dy = sh_y - ear_y
    angle_deg = math.degrees(math.atan2(dy, dx))
    if angle_deg < 0:
        angle_deg += 180
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    return round(angle_deg, 1), dx, dy

def generate_cervical_spine_3d(vec, cva_angle):
    dx, dy, dz = vec[0], vec[1], vec[2]
    names = ["C7", "C6", "C5", "C4", "C3", "C2 (Axis)", "C1 (Atlas)"]
    t_vals = np.linspace(0, 1, 7)
    curve_factor = 25.0 if cva_angle >= 50 else (45.0 - (cva_angle * 0.4))
    
    spine_x, spine_y, spine_z = [], [], []
    for t in t_vals:
        lx = t * dx
        ly = t * dz
        lz = t * dy
        offset = curve_factor * math.sin(t * math.pi) * (1 if dx >= 0 else -1)
        spine_x.append(lx + offset * 0.3)
        spine_y.append(ly)
        spine_z.append(lz)
        
    return spine_x, spine_y, spine_z, names

def create_interactive_3d_spine_plot(vec, cva_angle):
    dx, dy, dz = vec[0], vec[1], vec[2]
    u_len = max(abs(dx), 110) * (1 if dx >= 0 else -1)
    
    fig = go.Figure()
    sp_x, sp_y, sp_z, sp_names = generate_cervical_spine_3d(vec, cva_angle)

    # 1. เส้นแนวกระดูกคอ C1–C7
    fig.add_trace(go.Scatter3d(
        x=sp_x, y=sp_y, z=sp_z,
        mode='lines',
        line=dict(color='#cbd5e1', width=8),
        name="แนวกระดูกคอ (Cervical Alignment)"
    ))

    # 2. ข้อกระดูกสันหลัง C1 ถึง C7
    colors = ['#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16', '#10b981', '#06b6d4']
    fig.add_trace(go.Scatter3d(
        x=sp_x, y=sp_y, z=sp_z,
        mode='markers+text',
        marker=dict(size=9, color=colors, symbol='circle', line=dict(color='#0f172a', width=2)),
        text=sp_names,
        textposition="top right",
        textfont=dict(size=12, color='#ffffff'),
        name="ข้อกระดูก C1 - C7"
    ))

    # 3. เวกเตอร์ระนาบอ้างอิง u
    fig.add_trace(go.Scatter3d(
        x=[0, u_len], y=[0, 0], z=[0, 0],
        mode='lines+text',
        line=dict(color='#22c55e', width=6),
        text=["", "เวกเตอร์ u"],
        textposition="bottom center",
        name="เวกเตอร์ระนาบ u"
    ))
    fig.add_trace(go.Cone(
        x=[u_len], y=[0], z=[0],
        u=[u_len * 0.15], v=[0], w=[0],
        colorscale=[[0, '#22c55e'], [1, '#22c55e']],
        showscale=False, sizemode="absolute", sizeref=12, anchor="tip", name="หัวลูกศร u"
    ))

    # 4. เวกเตอร์กระดูกคอ v
    fig.add_trace(go.Scatter3d(
        x=[0, dx], y=[0, dz], z=[0, dy],
        mode='lines',
        line=dict(color='#fbbf24', width=5, dash='solid'),
        name="เวกเตอร์แนวคอรวม v"
    ))
    fig.add_trace(go.Cone(
        x=[dx], y=[dz], z=[dy],
        u=[dx * 0.15], v=[dz * 0.15], w=[dy * 0.15],
        colorscale=[[0, '#fbbf24'], [1, '#fbbf24']],
        showscale=False, sizemode="absolute", sizeref=14, anchor="tip", name="หัวลูกศร v"
    ))

    # 5. เส้นประโปรเจกต์ 3 มิติ
    fig.add_trace(go.Scatter3d(
        x=[dx, dx, 0],
        y=[dz, dz, 0],
        z=[dy, 0, 0],
        mode='lines',
        line=dict(color='#64748b', width=2, dash='dash'),
        name="เส้นฉายมิติ (dx, dy, dz)"
    ))

    max_val = max(abs(dx), abs(dy), abs(dz), 120) * 1.3
    fig.update_layout(
        title=f"🦴 แบบจำลองชีวกลศาสตร์ 3D: กระดูกคอ C1–C7 (CVA = {cva_angle}°)",
        scene=dict(
            xaxis=dict(title='X: แนวราบ (px)', range=[-max_val, max_val], backgroundcolor="rgba(15,23,42,0.4)"),
            yaxis=dict(title='Z: ความลึก (px)', range=[-max_val, max_val], backgroundcolor="rgba(15,23,42,0.4)"),
            zaxis=dict(title='Y: ความสูง (px)', range=[-20, max_val], backgroundcolor="rgba(15,23,42,0.4)"),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.35, y=-1.45, z=0.85))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

# ฟังก์ชันเขียนตัวหนังสือคมชัดสูง (มีเงาขอบดำลบรอยหยักด้วย cv2.LINE_AA)
def put_crisp_text(img, text, org, fontFace, fontScale, color, thickness=1, shadow_color=(0, 0, 0), shadow_thickness=3):
    # วาดเงาขอบสีดำรอบทิศทางเพื่อความคมชัด
    cv2.putText(img, text, org, fontFace, fontScale, shadow_color, shadow_thickness, cv2.LINE_AA)
    # วาดตัวหนังสือสีจริงทับด้านหน้า
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)

# ฟังก์ชันวาดป้ายข้อความพร้อมพื้นหลัง (Badge/Pill)
def draw_badge_label(img, text, org, text_color=(255, 255, 255), bg_color=(15, 23, 42), fontScale=0.48, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, fontScale, thickness)
    x, y = org
    pad_x, pad_y = 6, 4
    # กล่องพื้นหลังทึบ
    cv2.rectangle(img, (x - pad_x, y - th - pad_y), (x + tw + pad_x, y + baseline + pad_y - 2), bg_color, -1)
    # ขอบกล่องบาง
    cv2.rectangle(img, (x - pad_x, y - th - pad_y), (x + tw + pad_x, y + baseline + pad_y - 2), (71, 85, 105), 1, cv2.LINE_AA)
    # ตัวหนังสือ
    cv2.putText(img, text, (x, y), font, fontScale, text_color, thickness, cv2.LINE_AA)

# ฟังก์ชันวาดเส้นประใน OpenCV แบบคมชัด
def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=6):
    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist == 0:
        return
    pts = max(int(dist / gap), 1)
    for i in range(0, pts, 2):
        r1 = i / pts
        r2 = min((i + 1) / pts, 1.0)
        p1 = (int(pt1[0] + (pt2[0] - pt1[0]) * r1), int(pt1[1] + (pt2[1] - pt1[1]) * r1))
        p2 = (int(pt1[0] + (pt2[0] - pt1[0]) * r2), int(pt1[1] + (pt2[1] - pt1[1]) * r2))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

# ฟังก์ชันกลางสำหรับประมวลผลภาพและวาด HUD คมชัดระดับ HD
def analyze_and_draw_posture(img, pose_model):
    h, w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb_img)

    # 1. สร้างแถบ HUD คมชัดสูงแบบ High-Contrast Dark Theme
    hud_h = 118
    # วาดพื้นหลังแถบ HUD มืดสนิท 95% เพื่อตัดแสงสะท้อนและไม่ให้มี Noise
    hud_overlay = img.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (10, 15, 29), -1)
    img = cv2.addWeighted(hud_overlay, 0.94, img, 0.06, 0)
    
    # เส้นแบ่งล่างของแถบ HUD
    cv2.line(img, (0, hud_h), (w, hud_h), (56, 189, 248), 2, cv2.LINE_AA)
    
    # เส้นแบ่งกลางแนวตั้ง
    mid_x = w // 2
    cv2.line(img, (mid_x, 10), (mid_x, hud_h - 10), (51, 65, 85), 1, cv2.LINE_AA)

    cva_val, stat_val, vec_val, mag_val = None, None, None, None

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        r_ear = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR]
        r_sh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        l_ear = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR]
        l_sh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        
        if r_ear.visibility > l_ear.visibility:
            ear, sh = r_ear, r_sh
            x_dir = 1
            side_label = "SIDE: RIGHT"
        else:
            ear, sh = l_ear, l_sh
            x_dir = -1
            side_label = "SIDE: LEFT"
            
        if ear.visibility > 0.35 and sh.visibility > 0.35:
            ear_px, ear_py = int(ear.x * w), int(ear.y * h)
            sh_px, sh_py = int(sh.x * w), int(sh.y * h)
            
            cva_angle, dx, dy = calculate_cva_angle(ear_px, ear_py, sh_px, sh_py)
            dz = int((ear.z - sh.z) * w)
            mag_3d = round(math.sqrt(dx**2 + dy**2 + dz**2), 1)

            cva_val = cva_angle
            vec_val = [dx, dy, dz]
            mag_val = mag_3d

            p_c7 = (sh_px, sh_py)
            p_ear = (ear_px, ear_py)

            # กำหนดสถานะและสี
            if cva_angle >= 50:
                status_str, risk_str, color, stat_val = "NORMAL (Good Posture)", "Risk: LOW", (34, 197, 94), "ปกติ (Normal)"
            elif cva_angle >= 40:
                status_str, risk_str, color, stat_val = "MILD FORWARD HEAD", "Risk: MODERATE", (245, 158, 11), "คอยื่นเล็กน้อย (Mild FHP)"
            else:
                status_str, risk_str, color, stat_val = "SEVERE FORWARD HEAD", "Risk: HIGH (Text Neck)", (239, 68, 68), "คอยื่นรุนแรง (Severe FHP)"

            # === ฝั่งซ้ายของ HUD: ผลวิเคราะห์ทางการแพทย์ (CVA) ===
   
