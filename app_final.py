import streamlit as st
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# 저장 경로 설정
SAVE_DIR = "upload"
os.makedirs(SAVE_DIR, exist_ok=True)

# Streamlit UI 설정
st.image("image_logo.png")
st.title("대형 폐기물 이미지 분류")
st.write("이미지를 업로드하면, 해당 이미지가 어떤 클래스인지 분류해드립니다.♻️")
st.image("trash_image.png")

# 파일 업로드
uploaded_file = st.file_uploader("이미지 파일을 업로드하세요.", type=["jpg", "png", "jpeg"])

# 모델 로드 및 검증
# filepath = 'Efficient final.keras'
# model = load_model(filepath)

filepath = 'Efficient final.keras'  # 모델 경로를 확인
try:
    model = load_model(filepath)
    st.write("모델 로드 성공👍")
except Exception as e:
    st.error(f"모델 로드 실패👎: {e}")
    st.stop()

try:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.write("이미지 로드 성공👍")
except Exception as e:
    st.error(f"이미지 로드 실패👎: {e}")
    st.stop()

if uploaded_file is not None:
    st.write(f"파일 이름: {uploaded_file.name}")
    st.write(f"파일 타입: {uploaded_file.type}")
    st.write(f"파일 크기: {uploaded_file.size}, bytes")


    # 업로드된 이미지 표시
    st.image(uploaded_file, caption="업로드 이미지")


    # # 모델 구조 디버깅
    # st.write("모델 요약:")
    # model.summary()

    class_names = [
        'cabinet',
        'chair',
        'dining_table',
        'sofa'
    ]

    # 가격별 스티커 이미지 경로
    price_stickers = {
        'cabinet': ['sticker_3000.png', 'sticker_5000.png', 'sticker_10000.png'],
        'chair': ['sticker_3000.png', 'sticker_5000.png'],
        'dining_table': ['sticker_2000.png', 'sticker_3000.png'],
        'sofa': ['sticker_5000.png', 'sticker_10000.png']
    }

    IMAGE_SIZE = 224

    # 이미지 준비
    # image = Image.open(uploaded_file)
    # image_np = np.array(image)


    # 이미지 크기 조정 및 전처리
    resized_image = tf.image.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))

    # EfficientNet 전처리
    # a_image = np.array(resized_image)
    a_image = np.array(resized_image, dtype=np.float32)  # np.float32로 타입 명시
    a_image = preprocess_input(a_image)

    # batch_image = a_image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    batch_image = np.expand_dims(a_image, axis=0)  # 배치 차원 추가


    # # 디버깅 정보
    # st.write("이미지 전처리 정보:")
    # st.write(f"이미지 형태: {a_image.shape}")
    # st.write(f"이미지 최소/최대 값: {a_image.min()}, {a_image.max()}")

    # 예측
    pred_proba = model.predict(batch_image)

    # # 원시 예측 확률 출력
    # st.write("원시 예측 확률:")
    # st.write(pred_proba)

    # 최대 확률 클래스 선택
    pred = np.argmax(pred_proba)
    pred_label = class_names[pred]

    # 예측 결과 표시
    st.success(f"예측 클래스: {pred_label}")
    st.success(f"예측 확률: {pred_proba[0][pred] * 100:.2f}%")

    # 예측된 클래스에 해당하는 가격 스티커 선택
    sticker_choices = price_stickers[pred_label]  # 예측된 클래스에 해당하는 스티커 리스트
    selected_sticker_path = random.choice(sticker_choices)  # 랜덤으로 하나의 스티커 선택

    # 스티커 이미지 로드
    sticker = Image.open(selected_sticker_path)

    # 스티커 크기 조정 (원하는 크기로 조정)
    sticker = sticker.resize((100, 100))  # 예시로 크기 100x100으로 조정

    # 업로드된 이미지 위에 스티커 오버레이
    base_image = image.convert("RGBA")  # RGBA 모드로 변환 (투명 배경을 지원하기 위해)
    sticker = sticker.convert("RGBA")  # 스티커도 RGBA 모드로 변환

    # 스티커 위치 (업로드된 이미지에 맞춰서 위치 조정)
    sticker_position = (50, 50)  # 예시로 (50, 50) 위치에 스티커를 배치

    # # 이미지를 합성 (스티커가 업로드된 이미지 위에 덧씌워짐)
    combined = base_image.copy()
    combined.paste(sticker, sticker_position, sticker)  # 스티커 이미지가 투명 영역을 고려해서 합성

    # 최종 이미지 표시
    st.image(combined, caption="이미지와 폐기물 배출 스티커", use_column_width=True)

    # 가격 표시
    price = None
    if pred_label == 'cabinet':
        price = '3,000원' '5,000원' '10,000원'
    elif pred_label == 'chair':
        price = '3,000원' '5,000원'
    elif pred_label == 'dining_table':
        price = '2,000원' '3,000원'
    elif pred_label == 'sofa':
        price = '5,000원' '10,000원'

    st.write(f"해당 폐기물의 배출 스티커 예상 가격은 {price}입니다.")

    # 모델 로드는 위에서 진행하고, 이미지로드부터 이미지출력을 하나의 함수로 만들어서 출력완료시 다시 리턴할 수 있도록 만들기

    # # 각 클래스별 예측 확률
    # st.write("각 클래스에 대한 예측 확률:")
    # for i, class_name in enumerate(class_names):
    #     st.write(f"{class_name}: {pred_proba[0][i] * 100:.2f}%")
    #
    # # 예측 확률 막대 그래프
    # plt.figure(figsize=(10, 6))
    # plt.bar(class_names, pred_proba[0])
    # plt.title('예측 확률')
    # plt.xlabel('클래스')
    # plt.ylabel('확률')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # st.pyplot(plt)

