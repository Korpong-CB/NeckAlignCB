import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import mediapipe as mp
import av
import math
import os
import gc
import base64  # สำหรับฝังโลโก้ให้อยู่ในกรอบ HTML เดียวกัน 100%
from datetime import datetime
import numpy as np
import plotly.graph_objects as go

# 1. ตั้งค่าหน้าเว็บแบบ Wide
st.set_page_config(page_title="Neck Align CB", page_icon="📐", layout="wide")

# =========================================================================
# 1.1 CSS สไตล์ส่วนหัวพรีเมียม (รวมอยู่ในก้อน HTML เดียว ไม่หลุดกรอบ 100%)
# =========================================================================
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    
    .hero-card {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 45%, #1e293b 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 22px;
        padding: 26px 32px;
        margin-bottom: 18px;
        box-shadow: 0 12px 32px -10px rgba(2, 132, 199, 0.35), 0 0 20px 0 rgba(56, 189, 248, 0.1);
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        gap: 28px;
    }
    
    /* เส้นแสงนีออนไล่สีด้านบนกรอบ */
    .hero-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    }
    
    .hero-logo-box {
        flex-shrink: 0;
        width: 115px;
        height: 115px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.18) 0%, rgba(15, 23, 42, 0.75) 75%);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 12px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
    }
    
    .hero-logo-box img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
    
    .hero-content {
        flex-grow: 1;
    }
    
    .hero-brand {
        font-size: 34px !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
        background: linear-gradient(90deg, #38bdf8 0%, #60a5fa 45%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.3px;
    }
    
    .hero-sub1 {
        color: #f8fafc !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        margin: 0 0 4px 0 !important;
        line-height: 1.4 !important;
    }
    
    .hero-sub2 {
        color: #7dd3fc !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin: 0 0 10px 0 !important;
        line-height: 1.4 !important;
    }
    
    .hero-desc {
        color: #cbd5e1 !important;
        font-size: 14.5px !important;
        margin: 0 !important;
        padding-top: 8px;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
        line-height: 1.5;
    }
    
    .hero-desc b {
        color: #38bdf8;
    }
    
    @media (max-width: 768px) {
        .hero-card {
            flex-direction: column;
            text-align: center;
            padding: 20px;
            gap: 16px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Session State สำหรับเก็บประวัติรูปภาพ และค่าเวกเตอร์ 3D ล่าสุด
if "captured_records" not in st.session_state:
    st.session_state.captured_records = list()
if "active_vec" not in st.session_state:
    st.session_state.active_vec = list((75, 120, -20))
    st.session_state.active_cva = 55.0

# โหลดโมเดล MediaPipe สำหรับภาพนิ่ง
@st.cache_resource(max_entries=1)
def get_static_pose_model():
    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.4
    )

# ฟังก์ชันแปลงไฟล์ logo.png เป็น Base64 เพื่อฝังในกรอบ HTML เดียวกันโดยตรง
def get_logo_html():
    if os.path.exists("logo.png"):
        with open("logo.png", "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return f'<img src="data:image/png;base64,{b64_data}" alt="School Logo">'
    return '<span style="font-size: 56px;">🏫</span>'

# =========================================================================
# 2. ส่วนหัวโครงงานพรีเมียม (ฝังรวมใน HTML ก้อนเดียว กรอบตรงเป๊ะ 100%)
# =========================================================================
logo_html_str = get_logo_html()

st.markdown(f"""
    <div class="hero-card">
        <div class="hero-logo-box">
            {logo_html_str}
        </div>
        <div class="hero-content">
            <div class="hero-brand">Neck Align CB</div>
            <div class="hero-sub1">โครงงานคณิตศาสตร์ เรื่อง การประยุกต์ใช้เวกเตอร์สามมิติและฟังก์ชันตรีโกณมิติ</div>
            <div class="hero-sub1">เพื่อพัฒนาโปรแกรมประเมินมุม CVA เพื่อสังเกตภาวะศีรษะยื่นไปข้างหน้า (FHP)</div>
            <div class="hero-desc">
                ระบบตรวจวัดและวิเคราะห์มุมกระดูกสันหลังส่วนคอ (CVA) อัตโนมัติด้วย เวกเตอร์ 3 มิติ และฟังก์ชันตรีโกณมิติ พร้อมการจำลองแนวกระดูกคอ C1–C7 แบบเรียลไทม์
            </div>
        </div>
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
    dx, dy, dz = vec
    names = list(("C7", "C6", "C5", "C4", "C3", "C2 (Axis)", "C1 (Atlas)"))
    t_vals = np.linspace(0, 1, 7)
    curve_factor = 25.0 if cva_angle >= 50 else (45.0 - (cva_angle * 0.4))
    
    spine_x, spine_y, spine_z = list(), list(), list()
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
    dx, dy, dz = vec
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
    colors = list(('#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16', '#10b981', '#06b6d4'))
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
        x=list((0, u_len)), y=list((0, 0)), z=list((0, 0)),
        mode='lines+text',
        line=dict(color='#22c55e', width=6),
        text=list(("", "เวกเตอร์ u")),
        textposition="bottom center",
        name="เวกเตอร์ระนาบ u"
    ))
    fig.add_trace(go.Cone(
        x=list((u_len,)), y=list((0,)), z=list((0,)),
        u=list((u_len * 0.15,)), v=list((0,)), w=list((0,)),
        colorscale=list(((0, '#22c55e'), (1, '#22c55e'))),
        showscale=False, sizemode="absolute", sizeref=12, anchor="tip", name="หัวลูกศร u"
    ))

    # 4. เวกเตอร์กระดูกคอ v
    fig.add_trace(go.Scatter3d(
        x=list((0, dx)), y=list((0, dz)), z=list((0, dy)),
        mode='lines',
        line=dict(color='#fbbf24', width=5, dash='solid'),
        name="เวกเตอร์แนวคอรวม v"
    ))
    fig.add_trace(go.Cone(
        x=list((dx,)), y=list((dz,)), z=list((dy,)),
        u=list((dx * 0.15,)), v=list((dz * 0.15,)), w=list((dy * 0.15,)),
        colorscale=list(((0, '#fbbf24'), (1, '#fbbf24'))),
        showscale=False, sizemode="absolute", sizeref=14, anchor="tip", name="หัวลูกศร v"
    ))

    # 5. เส้นประโปรเจกต์แสดงองค์ประกอบ 3 มิติ
    fig.add_trace(go.Scatter3d(
        x=list((dx, dx, 0)),
        y=list((dz, dz, 0)),
        z=list((dy, 0, 0)),
        mode='lines',
        line=dict(color='#64748b', width=2, dash='dash'),
        name="เส้นฉายมิติ (dx, dy, dz)"
    ))

    max_val = max(abs(dx), abs(dy), abs(dz), 120) * 1.3
    fig.update_layout(
        title=f"🦴 แบบจำลองชีวกลศาสตร์ 3D: กระดูกคอ C1–C7 (CVA = {cva_angle}°)",
        scene=dict(
            xaxis=dict(title='X: แนวราบ (px)', range=list((-max_val, max_val)), backgroundcolor="rgba(15,23,42,0.4)"),
            yaxis=dict(title='Z: ความลึก (px)', range=list((-max_val, max_val)), backgroundcolor="rgba(15,23,42,0.4)"),
            zaxis=dict(title='Y: ความสูง (px)', range=list((-20, max_val)), backgroundcolor="rgba(15,23,42,0.4)"),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.35, y=-1.45, z=0.85))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

# ฟังก์ชันเขียนตัวหนังสือคมชัดสูง
def put_crisp_text(img, text, org, fontFace, fontScale, color, thickness=1, shadow_color=(0, 0, 0), shadow_thickness=3):
    cv2.putText(img, text, org, fontFace, fontScale, shadow_color, shadow_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)

# ฟังก์ชันวาดป้ายข้อความพร้อมพื้นหลัง
def draw_badge_label(img, text, org, text_color=(255, 255, 255), bg_color=(15, 23, 42), fontScale=0.48, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, fontScale, thickness)
    x, y = org
    pad_x, pad_y = 6, 4
    cv2.rectangle(img, (x - pad_x, y - th - pad_y), (x + tw + pad_x, y + baseline + pad_y - 2), bg_color, -1)
    cv2.rectangle(img, (x - pad_x, y - th - pad_y), (x + tw + pad_x, y + baseline + pad_y - 2), (71, 85, 105), 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, fontScale, text_color, thickness, cv2.LINE_AA)

# ฟังก์ชันวาดเส้นประใน OpenCV แบบคมชัด
def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=6):
    x1, y1 = pt1
    x2, y2 = pt2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    pts = max(int(dist / gap), 1)
    for i in range(0, pts, 2):
        r1 = i / pts
        r2 = min((i + 1) / pts, 1.0)
        p1 = (int(x1 + (x2 - x1) * r1), int(y1 + (y2 - y1) * r1))
        p2 = (int(x1 + (x2 - x1) * r2), int(y1 + (y2 - y1) * r2))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

# ฟังก์ชันกลางสำหรับประมวลผลภาพและวาด HUD คมชัดสูง (เกณฑ์ 3 ระดับ)
def analyze_and_draw_posture(img, pose_model, flip_h=True):
    if flip_h:
        img = cv2.flip(img, 1)
    h, w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb_img)

    hud_h = 118
    hud_overlay = img.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (10, 15, 29), -1)
    img = cv2.addWeighted(hud_overlay, 0.94, img, 0.06, 0)
    
    cv2.line(img, (0, hud_h), (w, hud_h), (56, 189, 248), 2, cv2.LINE_AA)
    
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
            vec_val = list((dx, dy, dz))
            mag_val = mag_3d

            p_c7 = (sh_px, sh_py)
            p_ear = (ear_px, ear_py)

            # 🟢🟡🔴 เกณฑ์การประเมินทางการแพทย์ 3 ระดับ
            if cva_angle > 53.0:
                if cva_angle > 55.0:
                    status_str, risk_str = "EXCELLENT POSTURE", "Severity: NONE (Excellent)"
                    stat_val = "ระดับปกติ (Normal/Excellent Posture)"
                else:
                    status_str, risk_str = "NORMAL POSTURE", "Severity: NONE (> 53 deg)"
                    stat_val = "ระดับปกติ (Normal Posture)"
                color = (50, 220, 50)       # 🟢 เขียว
            elif cva_angle >= 45.0:
                status_str, risk_str = "MODERATE SEVERITY", "Severity: MODERATE (45-52.9 deg)"
                color = (0, 215, 255)        # 🟡 เหลือง
                stat_val = "ภาวะคอยื่นระดับปานกลาง (Moderate)"
            else:
                status_str, risk_str = "SEVERE FORWARD HEAD", "Severity: SEVERE (< 45 deg)"
                color = (50, 50, 255)        # 🔴 แดง
                stat_val = "ภาวะคอยื่นระดับรุนแรง (Severe FHP)"

            # === ฝั่งซ้ายของ HUD ===
            put_crisp_text(img, f"CVA: {cva_angle:.1f} deg", (20, 38), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, shadow_thickness=4)
            put_crisp_text(img, status_str, (20, 74), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2, shadow_thickness=3)
            put_crisp_text(img, risk_str, (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, shadow_thickness=3)

            # === ฝั่งขวาของ HUD ===
            right_x = mid_x + 20
            put_crisp_text(img, f"3D Vector v = ({dx}, {dy}, {dz})", (right_x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2, shadow_thickness=3)
            put_crisp_text(img, f"||v|| = {mag_3d} px", (right_x, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (241, 245, 249), 1, shadow_thickness=3)
            put_crisp_text(img, side_label, (right_x, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (148, 163, 184), 1, shadow_thickness=3)

            # === วาดแนวแกนพิกัดฉาก 3 มิติ (X, Y, Z) ===
            axis_len = 80
            ax_x_x, ax_x_y = sh_px + int(axis_len * x_dir), sh_py
            ax_y_x, ax_y_y = sh_px, sh_py - axis_len
            ax_z_x, ax_z_y = sh_px - int(axis_len * 0.6 * x_dir), sh_py + int(axis_len * 0.4)

            cv2.arrowedLine(img, p_c7, (ax_x_x, ax_x_y), (80, 80, 240), 2, cv2.LINE_AA, tipLength=0.15)
            put_crisp_text(img, "X", (ax_x_x + 6 * x_dir, ax_x_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 240), 1)

            cv2.arrowedLine(img, p_c7, (ax_y_x, ax_y_y), (80, 240, 80), 2, cv2.LINE_AA, tipLength=0.15)
            put_crisp_text(img, "Y", (ax_y_x - 16, ax_y_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 240, 80), 1)

            cv2.arrowedLine(img, p_c7, (ax_z_x, ax_z_y), (240, 180, 60), 2, cv2.LINE_AA, tipLength=0.15)
            put_crisp_text(img, "Z", (ax_z_x - 16, ax_z_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 180, 60), 1)

            # เส้นประโปรเจกต์มิติสามมิติ
            p_floor = (ear_px, sh_py)
            draw_dashed_line(img, p_ear, p_floor, (220, 220, 220), 1, gap=5)
            draw_dashed_line(img, p_c7, p_floor, (140, 180, 220), 1, gap=5)

            # เวกเตอร์ฐาน u
            ref_x = sh_px + (125 * x_dir)
            cv2.arrowedLine(img, p_c7, (ref_x, sh_py), (50, 220, 50), 3, cv2.LINE_AA, tipLength=0.15)
            draw_badge_label(img, "u (1,0,0)", (ref_x - (10 if x_dir > 0 else 55), sh_py + 26), text_color=(50, 220, 50), bg_color=(15, 23, 42), fontScale=0.45)

            # เวกเตอร์คอ v
            cv2.arrowedLine(img, p_c7, p_ear, (0, 215, 255), 4, cv2.LINE_AA, tipLength=0.12)
            draw_badge_label(img, f"v ({dx},{dy},{dz})", (ear_px + 12, ear_py), text_color=(0, 215, 255), bg_color=(15, 23, 42), fontScale=0.45)

            # ส่วนโค้งองศามุม CVA
            start_angle = 0 if x_dir > 0 else 180
            end_angle = -int(cva_angle) if x_dir > 0 else 180 + int(cva_angle)
            cv2.ellipse(img, p_c7, (45, 45), 0, start_angle, end_angle, color, 2, cv2.LINE_AA)

            # วาดจุดข้อต่อร่างกายหลัก
            cv2.circle(img, p_c7, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, p_ear, 7, (255, 120, 0), -1, cv2.LINE_AA)
            cv2.circle(img, p_floor, 4, (180, 180, 180), -1, cv2.LINE_AA)
            
            # ป้ายบอกตำแหน่งจุดกำเนิด C7
            draw_badge_label(img, "C7 Origin (0,0,0)", (sh_px - 45, sh_py + 34), text_color=(255, 255, 255), bg_color=(15, 23, 42), fontScale=0.42)

        else:
            put_crisp_text(img, "Please turn side to camera clearly", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2, shadow_thickness=4)
    else:
        put_crisp_text(img, "Detecting posture... Please step into frame", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, shadow_thickness=4)

    return img, cva_val, stat_val, vec_val, mag_val

# 4. คลาส VideoProcessor สำหรับประมวลผลกล้องเรียลไทม์ WebRTC
class VideoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.latest_frame = None
        self.latest_cva = None
        self.latest_status = None
        self.latest_vec = None
        self.latest_mag = None
        self.is_front_cam = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, cva_val, stat_val, vec_val, mag_val = analyze_and_draw_posture(img, self.pose, flip_h=self.is_front_cam)

        self.latest_frame = img.copy()
        self.latest_cva = cva_val
        self.latest_status = stat_val
        self.latest_vec = vec_val
        self.latest_mag = mag_val

        del img
        gc.collect()

        return av.VideoFrame.from_ndarray(self.latest_frame, format="bgr24")

# 5. การจัดวางปุ่มและกล้องแบ่งเป็น 2 ฝั่ง คอลัมน์ซ้าย และขวา
col_cam, col_info = st.columns((3, 2))

with col_cam:
    st.markdown("### 📸 ระบบรับภาพและคำนวณเวกเตอร์ 3D")
    
    tab_live, tab_upload = st.tabs(list((
        "📹 โหมดกล้องเรียลไทม์ (Live Camera)", 
        "📱 โหมดอัปโหลดภาพถ่าย (สำหรับ iPad & มือถือ)"
    )))

    with tab_live:
        col_cam_mode, col_cam_tip = st.columns((1, 1))
        with col_cam_mode:
            cam_choice = st.radio(
                "📷 เลือกกล้องที่ใช้งาน:",
                options=list(("กล้องหน้า (Front Camera)", "กล้องหลัง (Back Camera)")),
                horizontal=True
            )
            is_front = (cam_choice == "กล้องหน้า (Front Camera)")
            video_constraints = {
                "facingMode": "user" if is_front else {"exact": "environment"},
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15}
            }

        with col_cam_tip:
            st.caption("💡 *หากเปลี่ยนกล้องแล้วภาพสลับช้า ให้กดปุ่ม **STOP** แล้วกด **START** ใหม่อีกครั้งเพื่อรีเซ็ตเลนส์กล้อง*")

        ctx = webrtc_streamer(
            key=f"neck-align-stream-{'front' if is_front else 'back'}",  
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": list(({"urls": list(("stun:stun.l.google.com:19302",))},))},
            media_stream_constraints={"video": video_constraints, "audio": False},
            async_processing=True
        )
        
        if ctx.video_processor:
            ctx.video_processor.is_front_cam = is_front
        
        if st.button("📸 บันทึกภาพจากกล้องสดและอัปเดตโมเดล 3D", use_container_width=True, type="primary"):
            if ctx.video_processor and ctx.video_processor.latest_frame is not None:
                vp = ctx.video_processor
                if vp.latest_cva is not None:
                    record = {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "img": cv2.cvtColor(vp.latest_frame, cv2.COLOR_BGR2RGB),
                        "cva": vp.latest_cva,
                        "status": vp.latest_status,
                        "vec": vp.latest_vec,
                        "mag": vp.latest_mag
                    }
                    st.session_state.captured_records.insert(0, record)
                    st.session_state.active_vec = vp.latest_vec
                    st.session_state.active_cva = vp.latest_cva
                    
                    if len(st.session_state.captured_records) > 3:
                        st.session_state.captured_records.pop()
                        
                    st.toast("✅ บันทึกภาพและจำลองแนวกระดูกคอ C1–C7 สำเร็จ!", icon="🦴")
                    gc.collect()
                else:
                    st.warning("⚠️ กล้องเปิดอยู่แต่ยังตรวจไม่พบมุมคอ กรุณาหันข้างเข้าหากล้องให้ชัดเจน")
            else:
                st.warning("⚠️ กรุณากดปุ่ม START เพื่อเปิดกล้องก่อนกดบันทึกภาพ")

    with tab_upload:
        st.info("💡 **วิธีใช้บน iPad:** กดปุ่ม **Browse files** ด้านล่าง -> เลือก **ถ่ายภาพ (Take Photo)** หรือ **คลังรูปภาพ (Photo Library)** เพื่อวิเคราะห์มุมคอทันที")
        uploaded_file = st.file_uploader(
            "📤 กดปุ่มเพื่อถ่ายภาพจากกล้อง iPad หรือเลือกรูปภาพด้านข้างจากอัลบั้ม", 
            type=list(("jpg", "jpeg", "png", "webp"))
        )
        
        if uploaded_file is not None:
            bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            cv2_img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
            
            if cv2_img is not None:
                max_dim = 800
                h_orig, w_orig, _ = cv2_img.shape
                if max(h_orig, w_orig) > max_dim:
                    scale = max_dim / max(h_orig, w_orig)
                    cv2_img = cv2.resize(cv2_img, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)

                pose_model = get_static_pose_model()
                processed_img, cva_val, stat_val, vec_val, mag_val = analyze_and_draw_posture(cv2_img, pose_model, flip_h=False)
                
                if cva_val is not None:
                    st.session_state.active_vec = vec_val
                    st.session_state.active_cva = cva_val
                    
                    rgb_out = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    st.image(rgb_out, caption=f"✅ วิเคราะห์สำเร็จ: มุม CVA = {cva_val}° ({stat_val})", use_column_width=True)
                    
                    if st.button("💾 บันทึกผลวิเคราะห์นี้ลงรายการเปรียบเทียบด้านล่าง", type="primary", use_container_width=True):
                        record = {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "img": rgb_out,
                            "cva": cva_val,
                            "status": stat_val,
                            "vec": vec_val,
                            "mag": mag_val
                        }
                        st.session_state.captured_records.insert(0, record)
                        
                        if len(st.session_state.captured_records) > 3:
                            st.session_state.captured_records.pop()
                            
                        st.toast("✅ บันทึกภาพและจำลองแนวกระดูกคอ C1–C7 สำเร็จ!", icon="🦴")
                        gc.collect()
                else:
                    st.error("⚠️ ตรวจไม่พบจุดหูและไหล่ที่ชัดเจน กรุณาใช้ภาพถ่ายที่หันด้านข้างให้เห็นช่วงศีรษะและหัวไหล่ชัดเจน")
                
                del cv2_img
                gc.collect()

    # พื้นที่แสดงผลประวัติแกลเลอรีภาพเปรียบเทียบสรีระ
    if len(st.session_state.captured_records) > 0:
        st.markdown("---")
        col_gal_title, col_gal_del = st.columns((3, 1))
        with col_gal_title:
            st.markdown(f"#### 📋 ภาพที่บันทึกผล (แสดงสูงสุด {len(st.session_state.captured_records)} รายการล่าสุด)")
        with col_gal_del:
            if st.button("🗑️ ล้างประวัติ", use_container_width=True):
                st.session_state.captured_records = list()
                st.cache_resource.clear()
                st.rerun()

        cols_snap = st.columns(min(len(st.session_state.captured_records), 2))
        for idx, rec in enumerate(st.session_state.captured_records):
            target_col = cols_snap[idx % 2]
            with target_col:
                with st.container(border=True):
                    st.image(rec.get("img"), use_column_width=True)
                    st.markdown(f"⏱️ **เวลา:** `{rec.get('time')}` | **CVA:** `{rec.get('cva')}°`")
                    st.markdown(f"🩺 **ผล:** `{rec.get('status')}`")
                    st.caption(f"📐 3D v: {rec.get('vec')} | ||v||: {rec.get('mag')} px")

with col_info:
    st.markdown("### 🦴 แบบจำลองกระดูกสันหลังส่วนคอ 3 มิติ (C1–C7)")
    st.caption("💡 *คลิกเมาส์ค้างหรือใช้นิ้วแตะเพื่อหมุนดูแนวกระดูก C1-C7 และเวกเตอร์ u, v ได้รอบทิศทาง 360°*")
    with st.container(border=True):
        fig_spine = create_interactive_3d_spine_plot(st.session_state.active_vec, st.session_state.active_cva)
        st.plotly_chart(fig_spine, use_container_width=True)

    # 📊 เกณฑ์การประเมินทางการแพทย์ (CVA) 3 ระดับตามตารางล่าสุด
    st.markdown("### 📊 เกณฑ์การประเมินทางการแพทย์ (CVA)")
    st.success("🟢 **ระดับปกติ (Normal Posture): > 53°**\n\nแนวกระดูกคอเรียงตัวตามธรรมชาติ สมดุลสรีระดีเยี่ยม (หากได้มุมมากกว่า 55° จะถือเป็นเกณฑ์สรีระปกติสมบูรณ์ หรือ Normal/Excellent Posture)")
    st.warning("🟡 **ภาวะคอยื่นระดับปานกลาง (Moderate Severity): 45° - 52.9°**\n\nพบคอยื่นระดับปานกลาง กล้ามเนื้อสะบัก บ่า และคอเริ่มตึงตัว ควรเริ่มบริหารยืดเหยียดกล้ามเนื้อ")
    st.error("🔴 **ภาวะคอยื่นระดับรุนแรง (Severe FHP / Text Neck): < 45°**\n\nพบคอยื่นระดับรุนแรง ศีรษะยื่นล้ำแนวไหล่อย่างชัดเจน กระดูกคอรับภาระน้ำหนักเสมือนสูง เสี่ยงต่อหมอนรองกระดูกคอเสื่อม")

st.markdown("---")

col_students, col_advisors = st.columns(2)
with col_students:
    with st.container(border=True):
        st.markdown("#### 👥 คณะผู้จัดทำโครงงาน")
        st.markdown("""
        1. **นางสาวทัดหทัย เดชหัสดิน** ชั้น ม.5/13
        2. **นายเมธาสิทธิ์ สัมฤทธิ์** ชั้น ม.5/13
        3. **นายปณพนธ์ วิระรังษิยากรณ์** ชั้น ม.5/13
        """)

with col_advisors:
    with st.container(border=True):
        st.markdown("#### 👨‍🏫 คณะครูที่ปรึกษาโครงงาน")
        st.markdown("""
        1. **นางสาวกัญจณ์วรา โนราช**
        2. **นายก่อพงศ์ ภูมิภักดี**
        """)

with st.container(border=True):
    st.markdown("<center>กลุ่มสาระการเรียนรู้คณิตศาสตร์ • โรงเรียนชัยภูมิภักดีชุมพล</center>", unsafe_allow_html=True)
