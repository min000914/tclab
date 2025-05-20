import streamlit as st
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import time
from random import randint, uniform

try:
    from tclab import setup, TCLab
    TCLAB_AVAILABLE = True
except ImportError:
    TCLAB_AVAILABLE = False

from src.policy import GaussianPolicy
from src.value_functions import TwinQ, ValueFunction
from src.iql import ImplicitQLearning
from src.util import torchify

#matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

MODEL_PATH  = "C:\\Users\\Developer\\TCLab\\IQL\\src\\offline\\best.pt"

st.set_page_config(page_title="TCLab 제어 대시보드", layout="wide")

st.title("🌡️ TCLab ‑ IQL 실시간 제어")

mode = st.radio("🧪 실행 환경", ["Simulator", "Real Kit"], horizontal=True)

dt        = 1.0
horizon_s = 1200
steps     = int(horizon_s / dt)        
obs_dim   = 4
act_dim   = 2

graph_type = st.selectbox("🎯 TSP 생성 방식", [ "랜덤","사용자지정", "사인"])
show_preview = st.checkbox("📈 TSP 미리보기", value=True)
x_axis_duration = st.number_input("Set X-axis Duration (seconds):", min_value=300, max_value=1500, value=1200)

x_values = np.arange(0, x_axis_duration, 1)
Tsp1=[]
Tsp2=[]
# 1️⃣ **랜덤 그래프**
if graph_type == "랜덤":
    st.write("### Random Graph Settings")

    # 🔹 **Temp 1 - 구간 분포 설정 (구간 길이 랜덤 분포)**
    mean_duration = st.number_input("Mean Section Duration (seconds):", min_value=1, max_value=x_axis_duration, value=int(x_axis_duration*0.4))
    std_duration = st.number_input("Std Deviation of Section Duration:", min_value=0, max_value=200, value=int(x_axis_duration*25//300))
    min_duration = st.number_input("Minimum Section Duration (seconds):", min_value=1, max_value=200, value=int(x_axis_duration*40//300))
    max_duration = st.number_input("Maximum Section Duration (seconds):", min_value=5, max_value=1000, value=int(x_axis_duration*200//300))
    temp_min = st.number_input("Min Temperature (°C):", min_value=0, max_value=100, value=25)
    temp_max = st.number_input("Max Temperature (°C):", min_value=0, max_value=100, value=65)


    if st.button("Generate Random Graph"):
        current_time_1, current_time_2 = 0, 0
        Tsp1, Tsp2 = [], []

        # 🔹 **Temp 1 Duration 및 온도 설정**
        while current_time_1 < x_axis_duration:
            duration_1 = int(np.clip(np.random.normal(mean_duration, std_duration), min_duration, max_duration))
            if current_time_1 + duration_1 > x_axis_duration:
                duration_1 = x_axis_duration - current_time_1

            target_temp1 = uniform(temp_min, temp_max)
            Tsp1.extend([target_temp1] * duration_1)
            current_time_1 += duration_1

        Tsp1.extend([Tsp1[-1]] * (x_axis_duration - len(Tsp1)))
        Tsp1 = np.array(Tsp1)

        # 🔹 **Temp 2 Duration 및 온도 설정**
        while current_time_2 < x_axis_duration:
            duration_2 = int(np.clip(np.random.normal(mean_duration, std_duration), min_duration, max_duration))
            if current_time_2 + duration_2 > x_axis_duration:
                duration_2 = x_axis_duration - current_time_2

            target_temp2 = uniform(temp_min, temp_max)
            Tsp2.extend([target_temp2] * duration_2)
            current_time_2 += duration_2

        Tsp2.extend([Tsp2[-1]] * (x_axis_duration - len(Tsp2)))
        Tsp2 = np.array(Tsp2)

# 2️⃣ **커스텀 그래프**
elif graph_type == "사용자지정":
    st.write("### Custom Graph Settings")
    sections = st.slider("Number of sections:", min_value=1, max_value=10, value=3)

    Tsp1, Tsp2 = [], []

    for i in range(sections):
        with st.expander(f"Section {i + 1} Settings", expanded=(i == 0)):
            st.write(f"#### Section {i + 1}")
            cols = st.columns(2)

            with cols[0]:
                st.write("**Temp 1 Settings (Blue)**")
                duration_1 = st.number_input(f"Duration (s)", min_value=0, max_value=x_axis_duration, value=x_axis_duration // sections, key=f"d1_{i}")
                temp1 = st.number_input(f"Target Temperature (°C)", min_value=0, max_value=100, value=25, key=f"t1_{i}")

            with cols[1]:
                st.write("**Temp 2 Settings (Red)**")
                duration_2 = st.number_input(f"Duration (s)", min_value=0, max_value=x_axis_duration, value=x_axis_duration // sections, key=f"d2_{i}")
                temp2 = st.number_input(f"Target Temperature (°C)", min_value=0, max_value=100, value=30, key=f"t2_{i}")
                
            # 값 추가
            Tsp1.extend([temp1] * duration_1)
            Tsp2.extend([temp2] * duration_2)

    # 길이 조정
    Tsp1 = np.array(Tsp1)[:x_axis_duration]
    Tsp2 = np.array(Tsp2)[:x_axis_duration]

# 3️⃣ **Sin 그래프**
elif graph_type == "Sin":
    st.write("### Sin Graph Settings")
    
    # 🔹 Amplitude and Frequency 설정
    amplitude1 = st.slider("Amplitude for Temp 1 (°C):", min_value=1, max_value=50, value=20)
    frequency1 = st.slider("Frequency for Temp 1 (Hz):", min_value=1, max_value=10, value=2)
    amplitude2 = st.slider("Amplitude for Temp 2 (°C):", min_value=1, max_value=50, value=15)
    frequency2 = st.slider("Frequency for Temp 2 (Hz):", min_value=1, max_value=10, value=3)
    
    # 🔹 Offset 설정 추가
    offset1 = st.number_input("Starting Value for Temp 1 (°C):", min_value=0, max_value=100, value=40)
    offset2 = st.number_input("Starting Value for Temp 2 (°C):", min_value=0, max_value=100, value=40)

    # 🔹 Sinusoidal Graph 생성
    Tsp1 = amplitude1 * np.sin(2 * np.pi * frequency1 * (x_values / x_axis_duration)) + offset1
    Tsp2 = amplitude2 * np.sin(2 * np.pi * frequency2 * (x_values / x_axis_duration)) + offset2


if show_preview and Tsp1 is not None:
    st.subheader("TSP 미리보기")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(Tsp1, label="TSP1")
    ax.plot(Tsp2, label="TSP2")
    ax.set_xlabel("Step"); ax.set_ylabel("°C")
    ax.grid(); ax.legend()
    st.pyplot(fig)

run = st.button("🚀 제어 시작")

@st.cache_resource
def load_iql_policy():
    policy = GaussianPolicy(obs_dim, act_dim, 256, 2)
    qf     = TwinQ(obs_dim, act_dim, 256, 2)
    vf     = ValueFunction(obs_dim, 256, 2)
    dummy_opt = lambda p: torch.optim.Adam(p, lr=1e-3)
    iql = ImplicitQLearning(qf, vf, policy, dummy_opt, max_steps=7500, tau=0.8, beta=3.0, alpha=0.005, discount=0.99)
    iql.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return iql.policy.eval()

if run and Tsp1 is not None:
    policy= load_iql_policy()

    if mode == "Simulator":
        if not TCLAB_AVAILABLE:
            st.error("tclab 패키지가 설치되지 않았습니다. Simulator 모드를 사용할 수 없습니다.")
            st.stop()
        lab_cls = setup(connected=False)
        env = lab_cls(synced=False)
    else:
        if not TCLAB_AVAILABLE:
            st.error("TCLab 하드웨어 라이브러리가 없습니다.")
            st.stop()
        try:
            env = TCLab()
        except Exception as ex:
            st.error(f"TCLab 연결 오류: {ex}")
            st.stop()

    env.Q1(0); env.Q2(0)
    if hasattr(env, "_T1"):  
        env._T1 = env._T2 = 29.0

    T1_list, T2_list, Q1_list, Q2_list = [], [], [], []
    total_ret = 0.0
    E1 = E2 = Over = Under = 0.0

    prog = st.progress(0.0)
    live_plot = st.empty()

    for k in range(steps):
        now = k * dt
        if hasattr(env, "update"):
            env.update(t=now)

        T1, T2 = env.T1, env.T2
        obs = torchify(np.array([T1, T2, Tsp1[k], Tsp2[k]], dtype=np.float32))
        with torch.no_grad():
            action = policy.act(obs, deterministic=True).cpu().numpy()
        Q1 = float(np.clip(action[0], 0.0, 100.0))
        Q2 = float(np.clip(action[1], 0.0, 100.0))
        env.Q1(Q1); env.Q2(Q2)

        T1_list.append(T1); T2_list.append(T2)
        Q1_list.append(Q1); Q2_list.append(Q2)

        err1 = Tsp1[k] - T1
        err2 = Tsp2[k] - T2
        raw_r = -np.sqrt(err1**2 + err2**2)
        total_ret += raw_r
        E1 += abs(err1); E2 += abs(err2)
        Over  += max(0, -err1) + max(0, -err2)
        Under += max(0,  err1) + max(0,  err2)

        if k % 5 == 0 or k == steps - 1:
            df_tmp = pd.DataFrame({"T1": T1_list, "T2": T2_list, "TSP1": Tsp1[:k+1], "TSP2": Tsp2[:k+1]})
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(df_tmp["T1"], label="T1"); ax.plot(df_tmp["T2"], label="T2")
            ax.plot(df_tmp["TSP1"], "--", label="TSP1"); ax.plot(df_tmp["TSP2"], ":", label="TSP2")
            ax.set_ylabel("°C"); ax.set_xlabel("Step"); ax.grid(); ax.legend(ncol=4, fontsize=8)
            live_plot.pyplot(fig)

        prog.progress((k+1) / steps)
        time.sleep(dt if mode == "Real Kit" else 0.01)

    env.Q1(0); env.Q2(0)

    st.subheader("✅ 제어 완료")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Return", f"{total_ret:.2f}")
    c2.metric("Total Error", f"{E1+E2:.2f}")
    c3.metric("Over | Under", f"{Over:.1f} / {Under:.1f}")

    df_out = pd.DataFrame({
        "T1": T1_list, "T2": T2_list,
        "Q1": Q1_list, "Q2": Q2_list,
        "TSP1": Tsp1, "TSP2": Tsp2
    })
    st.download_button("📥 CSV 다운로드", df_out.to_csv(index=False).encode("utf-8"), file_name="rollout")