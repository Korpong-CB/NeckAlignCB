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

# ฟังก์ชันวาดเส้นประใน OpenCV
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

# ฟังก์ชันกลางสำหรับประมวลผลภาพและวาด HUD 3D (ใช้ร่วมกันทั้งโหมด iPad และโหมดสตรีมสด)
def analyze_and_draw_posture(img, pose_model):
    h, w, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb_img)
    
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 115), (15, 23, 42), -1)
    img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
    
    mid_x = w // 2
    cv2.line(img, (mid_x, 10), (mid_x, 105), (70, 80, 95), 1, cv2.LINE_AA)

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
            
        if ear.visibility > 0.4 and sh.visibility > 0.4:
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

            # แกน X, Y, Z ที่จุด C7
            axis_len = 80
            ax_x = (sh_px + int(axis_len * x_dir), sh_py)
            ax_y = (sh_px, sh_py - axis_len)
            ax_z = (sh_px - int(axis_len * 0.6 * x_dir), sh_py + int(axis_len * 0.4))

            cv2.arrowedLine(img, p_c7, ax_x, (80, 80, 240), 2, cv2.LINE_AA, tipLength=0.15)
            cv2.putText(img, "X", (ax_x[0] + 5 * x_dir, ax_x[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 240), 1)
            cv2.arrowedLine(img, p_c7, ax_y, (80, 240, 80), 2, cv2.LINE_AA, tipLength=0.15)
            cv2.putText(img, "Y", (ax_y[0] - 15, ax_y[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 80), 1)
            cv2.arrowedLine(img, p_c7, ax_z, (240, 180, 60), 2, cv2.LINE_AA, tipLength=0.15)
            cv2.putText(img, "Z", (ax_z[0] - 15, ax_z[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 180, 60), 1)

            # เส้นประแสดงองค์ประกอบพิกัด
            p_floor = (ear_px, sh_py)
            draw_dashed_line(img, p_ear, p_floor, (220, 220, 220), 1, gap=5)
            draw_dashed_line(img, p_c7, p_floor, (140, 180, 220), 1, gap=5)

            # เวกเตอร์ u และ v
            ref_x = sh_px + (125 * x_dir)
            cv2.arrowedLine(img, p_c7, (ref_x, sh_py), (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.15)
            cv2.putText(img, "u", (ref_x - 10, sh_py + 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            cv2.arrowedLine(img, p_c7, p_ear, (0, 215, 255), 4, cv2.LINE_AA, tipLength=0.12)
            cv2.putText(img, "v", (ear_px + 12, ear_py), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 215, 255), 2)

            # ส่วนโค้งมุม CVA
            start_angle = 0 if x_dir > 0 else 180
            end_angle = -int(cva_angle) if x_dir > 0 else 180 + int(cva_angle)
            cv2.ellipse(img, p_c7, (45, 45), 0, start_angle, end_angle, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.circle(img, p_c7, 7, (0, 0, 255), -1)
            cv2.circle(img, p_ear, 7, (255, 120, 0), -1)
            cv2.circle(img, p_floor, 4, (180, 180, 180), -1)
            cv2.putText(img, "C7(0,0,0)", (p_c7[0] - 40, p_c7[1] + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            if cva_angle >= 50:
                status_str, risk_str, color, stat_val = "NORMAL (Good Posture)", "Risk: LOW", (50, 240, 50), "ปกติ (Normal)"
            elif cva_angle >= 40:
                status_str, risk_str, color, stat_val = "MILD FORWARD HEAD", "Risk: MODERATE", (0, 165, 255), "คอยื่นเล็กน้อย (Mild FHP)"
            else:
                status_str, risk_str, color, stat_val = "SEVERE FORWARD HEAD", "Risk: HIGH (Text Neck)", (50, 50, 255), "คอยื่นรุนแรง (Severe FHP)"

            cv2.putText(img, f"CVA: {cva_angle:.1f} deg", (15, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(img, status_str, (15, 72), cv2.FONT_HERSHEY_DUPLEX, 0.58, color, 2)
            cv2.putText(img, risk_str, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            right_x = mid_x + 15
            cv2.putText(img, f"3D Vector v = [{dx}, {dy}, {dz}]", (right_x, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2)
            cv2.putText(img, f"||v|| = {mag_3d} px", (right_x, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1)
            cv2.putText(img, side_label, (right_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (148, 163, 184), 1)
        else:
            cv2.putText(img, "Please turn side to camera clearly", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)
    else:
        cv2.putText(img, "Detecting posture... Please step into frame", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    return img, cva_val, stat_val, vec_val, mag_val

# 4. คลาส VideoProcessor สำหรับโหมด WebRTC
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

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img, cva_val, stat_val, vec_val, mag_val = analyze_and_draw_posture(img, self.pose)

        self.latest_frame = img.copy()
        self.latest_cva = cva_val
        self.latest_status = stat_val
        self.latest_vec = vec_val
        self.latest_mag = mag_val

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. การจัดวาง Layout แบ่ง 2 คอลัมน์
col_cam, col_info = st.columns([3, 2])

with col_cam:
    st.markdown("### 📹 ระบบกล้องตรวจจับและคำนวณเวกเตอร์ 3D")
    
    # แยก 2 แท็บ: แท็บแรกสำหรับ iPad/มือถือ (ติดชัวร์ 100%) และแท็บสองสำหรับกล้องสด WebRTC
    tab_ipad, tab_live = st.tabs([
        "📱 โหมดกล้องสำหรับ iPad / มือถือ (เสถียร 100%)", 
        "💻 โหมดกล้องวิดีโอสด (WebRTC Live)"
    ])

    with tab_ipad:
        st.info("💡 **วิธีใช้บน iPad:** หันด้านข้างเข้าหากล้อง แล้วกดปุ่ม **Take Photo** ระบบจะวาดเวกเตอร์ 3D และจำลองกระดูกคอ C1–C7 ทันที")
        img_file_buffer = st.camera_input("📸 กดปุ่มเพื่อถ่ายภาพวิเคราะห์มุมคอ (สามารถกดสลับกล้องหน้า-หลังได้ที่มุมขวาบนของจอภาพ)")
        
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.4) as static_pose:
                processed_img, cva_val, stat_val, vec_val, mag_val = analyze_and_draw_posture(cv2_img, static_pose)
                
                if cva_val is not None:
                    st.session_state.active_vec = vec_val
                    st.session_state.active_cva = cva_val
                    
                    rgb_out = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    st.image(rgb_out, caption=f"ผลการวิเคราะห์: มุม CVA = {cva_val}° ({stat_val})", use_column_width=True)
                    
                    if st.button("💾 บันทึกรูปนี้ลงแกลเลอรีด้านล่าง", type="primary", use_container_width=True):
                        record = {
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "img": rgb_out,
                            "cva": cva_val,
                            "status": stat_val,
                            "vec": vec_val,
                            "mag": mag_val
                        }
                        st.session_state.captured_records.insert(0, record)
                        st.toast("✅ บันทึกภาพและจำลองแนวกระดูกคอ C1–C7 สำเร็จ!", icon="🦴")
                else:
                    st.error("⚠️ ตรวจไม่พบจุดหูและไหล่ที่ชัดเจน กรุณากด Clear photo แล้วหันข้างถ่ายใหม่อีกครั้งครับ")

    with tab_live:
        ctx = webrtc_streamer(
            key="neck-align-stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {
                        "urls": [
                            "turn:openrelay.metered.ca:80",
                            "turn:openrelay.metered.ca:443",
                            "turn:openrelay.metered.ca:443?transport=tcp"
                        ],
                        "username": "openrelayproject",
                        "credential": "openrelayproject"
                    }
                ]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        
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
                    st.toast("✅ บันทึกภาพและจำลองแนวกระดูกคอ C1–C7 สำเร็จ!", icon="🦴")
                else:
                    st.warning("⚠️ กล้องเปิดอยู่แต่ยังตรวจไม่พบมุมคอ กรุณาหันข้างเข้าหากล้องให้ชัดเจนครับ")
            else:
                st.warning("⚠️ กรุณากดปุ่ม START เพื่อเปิดกล้องก่อนกดบันทึกภาพครับ")

    if len(st.session_state.captured_records) > 0:
        st.markdown("---")
        col_gal_title, col_gal_del = st.columns([3, 1])
        with col_gal_title:
            st.markdown(f"#### 📋 ภาพที่บันทึกผล ({len(st.session_state.captured_records)} รายการล่าสุด)")
        with col_gal_del:
            if st.button("🗑️ ล้างประวัติ", use_container_width=True):
                st.session_state.captured_records = []
                st.rerun()

        cols_snap = st.columns(min(len(st.session_state.captured_records), 2))
        for idx, rec in enumerate(st.session_state.captured_records):
            target_col = cols_snap[idx % 2]
            with target_col:
                with st.container(border=True):
                    st.image(rec["img"], use_column_width=True)
                    st.markdown(f"⏱️ **เวลา:** `{rec['time']}` | **CVA:** `{rec['cva']}°`")
                    st.markdown(f"🩺 **ผล:** `{rec['status']}`")
                    st.caption(f"📐 3D v: {rec['vec']} | ||v||: {rec['mag']} px")

with col_info:
    st.markdown("### 🦴 แบบจำลองกระดูกสันหลังส่วนคอ 3 มิติ (C1–C7)")
    st.caption("💡 *คลิกเมาส์ค้างหรือใช้นิ้วแตะเพื่อหมุนดูแนวกระดูก C1-C7 และเวกเตอร์ u, v ได้รอบทิศทาง 360°*")
    with st.container(border=True):
        fig_spine = create_interactive_3d_spine_plot(st.session_state.active_vec, st.session_state.active_cva)
        st.plotly_chart(fig_spine, use_container_width=True)

    st.markdown("### 📊 เกณฑ์การประเมินทางการแพทย์ (CVA)")
    st.success("🟢 **ตั้งแต่ 50.0° ขึ้นไป: ปกติ (Normal Posture)**\n\nแนวกระดูกคอสมดุล กล้ามเนื้อต้นคอรับภาระน้ำหนักศีรษะในเกณฑ์ปกติ (4.5 - 5.5 กก.)")
    st.warning("🟠 **40.0° - 49.9°: คอยื่นระดับเริ่มต้น (Mild FHP)**\n\nเริ่มมีอาการ Forward Head Posture กล้ามเนื้อบ่าและกระดูกคอเริ่มรับแรงเค้นเพิ่มขึ้น ควรยืดเหยียด")
    st.error("🔴 **ต่ำกว่า 40.0°: คอยื่นระดับรุนแรง (Severe FHP)**\n\nเสี่ยงต่อภาวะ Text Neck Syndrome กระดูกคอรับน้ำหนักเสมือนสูงถึง 18 - 27 กก. อาจเกิดอาการปวดเรื้อรัง")

    st.markdown("---")
    st.markdown("### 🧮 หลักการคณิตศาสตร์ที่ใช้ในโปรแกรม")
    with st.container(border=True):
        st.markdown("**1. ระบบพิกัดฉาก 3 มิติ (3D Coordinate Frame)**")
        st.caption("กำหนดจุดกระดูกคอ C7 เป็นจุดกำเนิด $O(0,0,0)$ มีแกน $X$ (แนวราบ), $Y$ (แนวดิ่ง), $Z$ (ความลึก)")
        st.markdown(r"**2. เวกเตอร์ระนาบอ้างอิง ($\vec{u}$) และเวกเตอร์แนวกระดูกคอ ($\vec{v}$)**")
        st.latex(r"\vec{u} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \vec{v} = \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta z \end{bmatrix} = \begin{bmatrix} x_{ear} - x_{c7} \\ y_{c7} - y_{ear} \\ z_{ear} - z_{c7} \end{bmatrix}")
        st.markdown(r"**3. ขนาดของเวกเตอร์ 3 มิติ (Magnitude $\|\vec{v}\|$)**")
        st.latex(r"\|\vec{v}\| = \sqrt{(\Delta x)^2 + (\Delta y)^2 + (\Delta z)^2}")
        st.markdown(r"**4. การคำนวณมุม Craniovertebral Angle (CVA) ด้วยตรีโกณมิติ**")
        st.latex(r"\theta = \arctan2(\Delta y, \Delta x) \times \left(\frac{180^\circ}{\pi}\right)")

st.markdown("---")
st.markdown("### 🎓 ข้อมูลโครงงานและคณะผู้จัดทำ")

col_students, col_advisors = st.columns(2)
with col_students:
    with st.container(border=True):
        st.markdown("#### 👥 นักเรียนผู้จัดทำโครงงาน")
        st.markdown("""
        1. **นางสาวทัดหทัย เดชหัสดิน** `ชั้น ม.5/13`
        2. **นายเมธาสิทธิ์ สัมฤทธิ์** `ชั้น ม.5/13`
        3. **นายปณพนธ์ วิระรังษิยากรณ์** `ชั้น ม.5/13`
        """)

with col_advisors:
    with st.container(border=True):
        st.markdown("#### 👨‍🏫 คณะครูที่ปรึกษาโครงงาน")
        st.markdown("""
        * 🔹 **นางสาวกัญจณ์วรา โนราช** *(ครูที่ปรึกษา)*
        * 🔹 **นายก่อพงศ์ ภูมิภักดี** *(ครูที่ปรึกษา)*
        """)

with st.container(border=True):
    st.markdown("<center>🏫 กลุ่มสาระการเรียนรู้คณิตศาสตร์ • <b>โรงเรียนชัยภูมิภักดีชุมพล</b></center>", unsafe_allow_html=True)
