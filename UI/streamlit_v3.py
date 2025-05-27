# Streamlit UI 리팩토링 - 설정 / 그래프 설정 영역을 세로선으로 구분, 그래프는 아래 중앙 배치

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
from random import uniform
import os, sys
from pathlib import Path

# 시스템 설정 및 모듈 불러오기
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tclab import setup, TCLab
from IQL.src.policy import GaussianPolicy, MPCBasedGaussianPolicy
from IQL.src.value_functions import TwinQ, ValueFunction
from IQL.src.iql import ImplicitQLearning
from IQL.src.util import torchify

# 페이지 기본 설정
st.set_page_config(page_title="TCLab 제어 대시보드", layout="wide")
st.title("🌡️ TCLab ‑ 실시간 제어")

MODEL_PATH_MPC = "UI/model/MPC_based_RL.pt"
MODEL_PATH_PID = "UI/model/PID_based_RL.pt"

# 설정과 그래프 설정 영역을 좌우로 나누고 가운데 세로선 삽입
col_left, col_divider, col_right = st.columns([1, 0.02, 2.5])

with col_left:
    st.header("🧩 그래프 설정")
    x_axis_duration = st.slider("X축 지속시간 (초)", 300, 1500, 1200)
    graph_type = st.selectbox("TSP 생성 방식", ["Random", "Custom", "Sinusoidal"])
    x_values = np.arange(0, x_axis_duration, 1)

    if 'Tsp1' not in st.session_state:
        st.session_state.Tsp1 = np.zeros_like(x_values, dtype=float)
    if 'Tsp2' not in st.session_state:
        st.session_state.Tsp2 = np.zeros_like(x_values, dtype=float)

    Tsp1 = st.session_state.Tsp1
    Tsp2 = st.session_state.Tsp2

    if graph_type == "Random":
        st.write("### Random Graph Settings")

        # 설정값
        mean_duration = st.number_input("Mean Section Duration (seconds):", min_value=1, max_value=x_axis_duration, value=int(x_axis_duration*0.4))
        std_duration = st.number_input("Std Deviation of Section Duration:", min_value=0, max_value=200, value=int(x_axis_duration*25//300))
        min_duration = x_axis_duration*40//300
        max_duration = x_axis_duration*200//300
        temp_min = st.number_input("Min Temperature (°C):", min_value=0, max_value=100, value=25)
        temp_max = st.number_input("Max Temperature (°C):", min_value=0, max_value=100, value=65)

        if st.button("Generate Random Graph"):
            # Temp 1
            current_time_1 = 0
            while current_time_1 < x_axis_duration:
                duration = int(np.clip(np.random.normal(mean_duration, std_duration), min_duration, max_duration))
                duration = min(duration, x_axis_duration - current_time_1)
                temp = uniform(temp_min, temp_max)
                Tsp1[current_time_1:current_time_1+duration] = temp
                current_time_1 += duration
            # Temp 2
            current_time_2 = 0
            while current_time_2 < x_axis_duration:
                duration = int(np.clip(np.random.normal(mean_duration, std_duration), min_duration, max_duration))
                duration = min(duration, x_axis_duration - current_time_2)
                temp = uniform(temp_min, temp_max)
                Tsp2[current_time_2:current_time_2+duration] = temp
                current_time_2 += duration
        temp_min = temp_min // 10 * 10
        temp_max = temp_max // 10 * 10 + 10
        
    elif graph_type == "Custom":
        st.write("### Custom Graph Settings")
        sections = st.slider("Number of sections:", min_value=1, max_value=10, value=3)

        idx_1, idx_2 = 0, 0
        for i in range(sections):
            with st.expander(f"Section {i + 1} Settings", expanded=(i == 0)):
                cols = st.columns(2)
                with cols[0]:
                    duration_1 = st.number_input(f"Duration (s)", min_value=0, max_value=x_axis_duration-idx_1, value= ((x_axis_duration-idx_1) // (sections-i)), key=f"d1_{i}")
                    temp1 = st.number_input(f"Target Temperature (°C)", min_value=0, max_value=100, value=25, key=f"t1_{i}")
                with cols[1]:
                    duration_2 = st.number_input(f"Duration (s)", min_value=0, max_value=x_axis_duration-idx_2, value=((x_axis_duration-idx_2) // (sections-i)), key=f"d2_{i}")
                    temp2 = st.number_input(f"Target Temperature (°C)", min_value=0, max_value=100, value=30, key=f"t2_{i}")

            Tsp1[idx_1:idx_1+duration_1] = temp1
            Tsp2[idx_2:idx_2+duration_2] = temp2
            idx_1 += duration_1
            idx_2 += duration_2
        
        temp_min= np.concatenate([Tsp1, Tsp2]).min()//10 *10
        temp_max= np.concatenate([Tsp1, Tsp2]).max()//10 *10 + 10

    elif graph_type == "Sinusoidal":
        st.write("### Sin Graph Settings")

        amplitude1 = st.slider("Amplitude for Temp 1 (°C):", min_value=1, max_value=50, value=20)
        frequency1 = st.slider("Frequency for Temp 1 (Hz):", min_value=1, max_value=10, value=2)
        amplitude2 = st.slider("Amplitude for Temp 2 (°C):", min_value=1, max_value=50, value=15)
        frequency2 = st.slider("Frequency for Temp 2 (Hz):", min_value=1, max_value=10, value=3)

        offset1 = st.number_input("Starting Value for Temp 1 (°C):", min_value=0, max_value=100, value=40)
        offset2 = st.number_input("Starting Value for Temp 2 (°C):", min_value=0, max_value=100, value=40)

        Tsp1[:] = amplitude1 * np.sin(2 * np.pi * frequency1 * (x_values / x_axis_duration)) + offset1
        Tsp2[:] = amplitude2 * np.sin(2 * np.pi * frequency2 * (x_values / x_axis_duration)) + offset2


        temp_min= np.concatenate([Tsp1, Tsp2]).min()//10 *10
        temp_max= np.concatenate([Tsp1, Tsp2]).max()//10 *10 + 10
        
    st.session_state.Tsp1 = Tsp1
    st.session_state.Tsp2 = Tsp2
    Tsp1 = st.session_state.Tsp1
    Tsp2 = st.session_state.Tsp2
    
    st.markdown("---")
    st.header("⚙️ 실행 설정")
    mode = st.radio("실행 환경", ["Simulator", "RealKit"], horizontal=True)
    model = st.radio("모델 선택", ["PID", "MPC", "희진", "창기"], horizontal=True)
    st_temp = 29

    # 연결 상태 세션 변수 초기화
    if 'env_connected' not in st.session_state:
        st.session_state.env_connected = False
        st.session_state.env = None
        st.session_state.last_mode = None

    # mode가 바뀌었거나 아직 연결되지 않았으면 연결 시도
    if (not st.session_state.env_connected) or (st.session_state.last_mode != mode):
        try:
            if mode == "RealKit":
                st.session_state.env = TCLab()
            else:
                lab = setup(connected=False)
                st.session_state.env = lab(synced=False)
            st.session_state.env_connected = True
            st.session_state.last_mode = mode
            st.success(f"{mode} 연결 완료")
        except Exception as e:
            st.session_state.env_connected = False
            st.session_state.env = None
            st.error(f"{mode} 연결 실패: {e}")

    # 연결된 환경 참조
    if st.session_state.env_connected:
        env = st.session_state.env
        st.info(f"✅ 연결됨: {mode}")
    else:
        st.warning("연결되지 않았습니다.")


    if model == "희진":
        MODEL_PATH = MODEL_PATH_MPC
        obs_dim = 4
    elif model == "창기":
        MODEL_PATH = MODEL_PATH_PID
        obs_dim = 6
    else:
        MODEL_PATH = None
        obs_dim = 0

    act_dim = 2
    dt = 1.0
    steps = int(x_axis_duration / dt)

with col_divider:
    st.markdown(
        """
        <div style='border-left:1px solid lightgray;height:150vh;position:relative;left:50%;'></div>
        """,
        unsafe_allow_html=True
    )
    
    
with col_right:
    # 아래 중앙에서 그래프 시각화
    st.subheader("📉 목표 온도 그래프")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x_values, st.session_state.Tsp1, label="TSP1 (Temp 1)", color='blue')
    ax.plot(x_values, st.session_state.Tsp2, label="TSP2 (Temp 2)", color='red')
    ax.set_xlabel("Step")
    ax.set_ylabel("°C")
    ax.set_title(f"{graph_type} Target Temperatures")
    ax.legend()
    st.pyplot(fig)

    run = st.button("🚀 제어 시작")

    def load_iql_policy():
        policy = MPCBasedGaussianPolicy(obs_dim, act_dim, 256, 2) if model == "희진" else GaussianPolicy(obs_dim, act_dim, 256, 2)
        qf = TwinQ(obs_dim, act_dim, 256, 2)
        vf = ValueFunction(obs_dim, 256, 2)
        dummy_opt = lambda p: torch.optim.Adam(p, lr=1e-3)
        iql = ImplicitQLearning(qf, vf, policy, dummy_opt, max_steps=7500, tau=0.8, beta=3.0, alpha=0.005, discount=0.99)
        iql.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        return iql.policy.eval()

    if run:
        T1_list, T2_list, Q1_list, Q2_list = [], [], [], []
        total_ret = E1 = E2 = Over = Under = 0.0
        prog = st.progress(0.0)
        live_plot1 = st.empty()
        live_plot2 = st.empty()
        env.Q1(0)
        env.Q2(0)
        if mode == "RealKit":
            while env.T1 >= st_temp or env.T2 >= st_temp:
                time.sleep(20)

            dt_error = 0.0
            start_time = time.time()
            prev_time = start_time
        
        if model in ["희진", "창기"]:
            policy = load_iql_policy()

        if model == "PID":
            # PID 상수 및 함수 정의
            Kc, tauI, tauD, Kff = 9.24, 126.6, 8.90, -0.66

            def pid(sp, pv, pv_last, ierr, dt, d, cid):
                KP, Kf = (Kc, Kff) if cid == 1 else (Kc * 2.0, Kff * 2.0)
                KI = Kc / tauI
                KD = Kc * tauD
                op0, oplo, ophi = 0, 0, 100
                error = sp - pv
                ierr += KI * error * dt
                dpv = (pv - pv_last) / dt
                P, I, D, FF = KP * error, ierr, -KD * dpv, Kf * d
                op = op0 + P + I + D + FF
                if op < oplo or op > ophi:
                    I -= KI * error * dt
                    op = max(oplo, min(ophi, op))
                return op, I

            ierr1 = ierr2 = 0.0

        for i in range(steps):
            if mode == "Simulator":
                now = i * dt
                env.update(t=now)
                
            elif mode == "RealKit":
                sleep = dt - (time.time() - prev_time) - dt_error
                if sleep >= 1e-4:
                    time.sleep(sleep - 1e-4)
                else:
                    print('exceeded max cycle time by ' + str(abs(sleep)) + ' sec')
                    time.sleep(1e-4)

                t = time.time()
                step_dt = t - prev_time
                if (sleep>=1e-4):
                    dt_error = step_dt-dt+0.009
                else:
                    dt_error = 0.0
                prev_time = t
                
            T1, T2 = env.T1, env.T2

            if model == "희진":
                obs = torchify(np.array([T1, T2, st.session_state.Tsp1[i], st.session_state.Tsp2[i]], dtype=np.float32))
                with torch.no_grad():
                    action = policy.act(obs, deterministic=True).cpu().numpy()
                Q1 = float(np.clip(action[0], 0, 100))
                Q2 = float(np.clip(action[1], 0, 100))

            elif model == "창기":
                if i == 0:
                    dT1 = dT2 = 0
                elif i < 4:
                    dT1 = T1 - T1_list[i-1]
                    dT2 = T2 - T2_list[i-1]
                else:
                    dT1 = T1 - T1_list[i-4]
                    dT2 = T2 - T2_list[i-4]
                obs = torchify(np.array([T1, st.session_state.Tsp1[i], dT1, T2, st.session_state.Tsp2[i], dT2], dtype=np.float32))
                with torch.no_grad():
                    action = policy.act(obs, deterministic=True).cpu().numpy()
                Q1 = float(np.clip(action[0], 0, 100))
                Q2 = float(np.clip(action[1], 0, 100))

            elif model == "PID":
                d1 = T1 - 23.0
                d2 = T2 - 23.0
                Q1, ierr1 = pid(st.session_state.Tsp1[i], T1, T1_list[i-1] if i > 0 else T1, ierr1, dt, d2, 1)
                Q2, ierr2 = pid(st.session_state.Tsp2[i], T2, T2_list[i-1] if i > 0 else T2, ierr2, dt, d1, 2)

            env.Q1(Q1); env.Q2(Q2)

            T1_list.append(T1); T2_list.append(T2)
            Q1_list.append(Q1); Q2_list.append(Q2)

            err1 = st.session_state.Tsp1[i] - T1
            err2 = st.session_state.Tsp2[i] - T2
            raw_r = -np.sqrt(err1**2 + err2**2)
            total_ret += raw_r
            E1 += abs(err1); E2 += abs(err2)
            Over += max(0, -err1) + max(0, -err2)
            Under += max(0, err1) + max(0, err2)

            if i % 5 == 0 or i == steps - 1:
                df_tmp = pd.DataFrame({"T1": T1_list, "T2": T2_list, "TSP1": st.session_state.Tsp1[:i+1], "TSP2": st.session_state.Tsp2[:i+1]})
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(df_tmp["T1"], label="T1"); ax.plot(df_tmp["T2"], label="T2")
                ax.plot(df_tmp["TSP1"], "--", label="TSP1"); ax.plot(df_tmp["TSP2"], ":", label="TSP2")
                ax.set_ylabel("°C"); ax.set_xlabel("Step"); ax.grid(); ax.legend(ncol=4, fontsize=8)
                live_plot1.pyplot(fig)
                
                df_tmp = pd.DataFrame({"Q1": Q1_list, "Q2": Q2_list})
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(df_tmp["Q1"], label="Q1"); ax.plot(df_tmp["Q2"], label="Q2")
                ax.set_ylabel("Q (0-100)"); ax.set_xlabel("Step"); ax.grid(); ax.legend(ncol=2, fontsize=8)
                live_plot2.pyplot(fig)
                
                plt.close(fig) 
            prog.progress((i+1) / steps)
            time.sleep(dt if mode == "Real Kit" else 0.01)

        env.Q1(0); env.Q2(0)

        if T1_list and T2_list:
            st.subheader("✅ 제어 완료")
            if env is not None:
                env.close()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Return", f"{total_ret:.2f}")
            c2.metric("Total Error", f"{E1+E2:.2f}")
            c3.metric("Over | Under", f"{Over:.1f} / {Under:.1f}")

            df_out = pd.DataFrame({
                "T1": T1_list, "T2": T2_list,
                "Q1": Q1_list, "Q2": Q2_list,
                "TSP1": st.session_state.Tsp1[:steps], "TSP2": st.session_state.Tsp2[:steps]
            })
            st.download_button("📥 CSV 다운로드", df_out.to_csv(index=False).encode("utf-8"), file_name="rollout.csv")
