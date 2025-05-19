import streamlit as st
from tclab import TCLab, TCLabModel
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint, uniform
import random

# 페이지 구성
st.title("TCLab Controller")
st.subheader("Choose a connection type:")

# 버튼 생성
if st.button("Use Simulator"):
    with st.spinner("Connecting to Simulator..."):
        try:
            lab = TCLabModel()
            time.sleep(2)  # 연결 대기 시간
            st.success("Simulator Connected Successfully!")
        except Exception as e:
            st.error(f"Failed to connect Simulator: {e}")

if st.button("Use TCLab Kit"):
    with st.spinner("Connecting to TCLab Kit..."):
        try:
            lab = TCLab()
            time.sleep(2)  # 연결 대기 시간
            st.success("TCLab Kit Connected Successfully!")
        except Exception as e:
            st.error(f"Failed to connect Kit: {e}")

# 그래프 타입 선택
graph_type = st.selectbox("Select Graph Type:", ["Random", "Custom", "Sin"])

# X축 설정
x_axis_duration = st.number_input("Set X-axis Duration (seconds):", min_value=300, max_value=1500, value=1200)

# 빈 데이터프레임 생성
x_values = np.arange(0, x_axis_duration, 1)
y_values_1 = np.zeros_like(x_values)
y_values_2 = np.zeros_like(x_values)

# 그래프 설정
fig, ax = plt.subplots()

# 1️⃣ **랜덤 그래프**
if graph_type == "Random":
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
        y_values_1, y_values_2 = [], []

        # 🔹 **Temp 1 Duration 및 온도 설정**
        while current_time_1 < x_axis_duration:
            duration_1 = int(np.clip(np.random.normal(mean_duration, std_duration), min_duration, max_duration))
            if current_time_1 + duration_1 > x_axis_duration:
                duration_1 = x_axis_duration - current_time_1

            target_temp1 = uniform(temp_min, temp_max)
            y_values_1.extend([target_temp1] * duration_1)
            current_time_1 += duration_1

        y_values_1.extend([y_values_1[-1]] * (x_axis_duration - len(y_values_1)))
        y_values_1 = np.array(y_values_1)

        # 🔹 **Temp 2 Duration 및 온도 설정**
        while current_time_2 < x_axis_duration:
            duration_2 = int(np.clip(np.random.normal(mean_duration, std_duration), min_duration, max_duration))
            if current_time_2 + duration_2 > x_axis_duration:
                duration_2 = x_axis_duration - current_time_2

            target_temp2 = uniform(temp_min, temp_max)
            y_values_2.extend([target_temp2] * duration_2)
            current_time_2 += duration_2

        y_values_2.extend([y_values_2[-1]] * (x_axis_duration - len(y_values_2)))
        y_values_2 = np.array(y_values_2)

# 2️⃣ **커스텀 그래프**
elif graph_type == "Custom":
    st.write("### Custom Graph Settings")
    sections = st.slider("Number of sections:", min_value=1, max_value=10, value=3)

    y_values_1, y_values_2 = [], []

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
            y_values_1.extend([temp1] * duration_1)
            y_values_2.extend([temp2] * duration_2)

    # 길이 조정
    y_values_1 = np.array(y_values_1)[:x_axis_duration]
    y_values_2 = np.array(y_values_2)[:x_axis_duration]

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
    y_values_1 = amplitude1 * np.sin(2 * np.pi * frequency1 * (x_values / x_axis_duration)) + offset1
    y_values_2 = amplitude2 * np.sin(2 * np.pi * frequency2 * (x_values / x_axis_duration)) + offset2


# 🔹 **그래프에 두 개의 온도 목표 표시**
ax.plot(x_values, y_values_1, label="Temp 1 (Blue)", color='blue')
ax.plot(x_values, y_values_2, label="Temp 2 (Red)", color='red')
ax.set_title(f"{graph_type} Target Temperatures")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Target Temperature (°C)")
ax.legend()

# 그래프 출력
st.pyplot(fig)
