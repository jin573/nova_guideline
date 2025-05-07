import cv2
import numpy as np
import os
import csv

# 휘도 계산 함수 (상대 휘도 계산)
def calculate_luminance(frame):
    """
    주어진 프레임에서 각 픽셀의 휘도를 계산합니다.
    휘도는 Y = 0.2126 * R + 0.7152 * G + 0.0722 * B 공식을 사용하여 계산됩니다.
    
    :param frame: BGR 형식의 이미지 프레임
    :return: 계산된 휘도 값 (같은 크기의 2D 배열)
    """
    return 0.2126 * frame[:, :, 2] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 0]

# 휘도 차이 계산 함수
def calculate_luminance_diff(curr_luminance, prev_luminance):
    """
    현재 프레임과 이전 프레임의 휘도 차이를 계산합니다.
    
    :param curr_luminance: 현재 프레임의 휘도 값
    :param prev_luminance: 이전 프레임의 휘도 값
    :return: 휘도 차이 (절댓값)
    """
    diff = np.abs(curr_luminance - prev_luminance)
    return diff

# 면적 비율을 계산하여 기준을 넘는지 체크하는 함수
def check_area_ratio(diff, prev_luminance, threshold, total_area, area_ratio):
    """
    휘도 차이가 기준 이상인 부분의 면적 비율을 계산하고,
    화면에서 일정 비율 이상인지를 확인합니다.
    
    :param diff: 휘도 차이 값
    :param prev_luminance: 이전 프레임의 휘도 값
    :param threshold: 휘도 차이 임계값
    :param total_area: 화면의 전체 면적
    :param area_ratio: 체크할 면적 비율 기준
    :return: 면적 비율이 기준 이상이면 True, 아니면 False
    """
    relative_diff = diff / (prev_luminance + 1e-6)  # 0으로 나누는 오류 방지
    mask = relative_diff >= threshold
    change_area = np.sum(mask)
    return (change_area / total_area) >= area_ratio

# 깜빡임 주파수를 검사하여 기준에 맞는 구간을 탐지하는 함수
def detect_flickering(flicker_frames, fps, min_flicker_hz, max_flicker_hz):
    """
    초당 깜빡임 횟수에 맞춰 프레임을 그룹화합니다.
    이 함수는 깜빡임 구간을 연속된 시간 간격으로 구분합니다.
    
    :param flicker_frames: 깜빡임이 일어난 프레임 번호 리스트
    :param fps: 초당 프레임 수 (비디오의 FPS)
    :param min_flicker_hz: 최소 깜빡임 주파수 (초당 몇 번)
    :param max_flicker_hz: 최대 깜빡임 주파수 (초당 몇 번)
    :return: 깜빡임이 감지된 구간 (시작 프레임, 끝 프레임)의 리스트
    """
    clip_ranges = []
    min_gap = int(fps / max_flicker_hz)
    max_gap = int(fps / min_flicker_hz)
    
    temp = []
    for i in range(len(flicker_frames)):
        if i == 0 or flicker_frames[i] - flicker_frames[i-1] <= max_gap:
            temp.append(flicker_frames[i])
        else:
            if len(temp) >= min_gap:
                clip_ranges.append((temp[0], temp[-1]))
            temp = [flicker_frames[i]]
    if len(temp) >= min_gap:
        clip_ranges.append((temp[0], temp[-1]))

    return clip_ranges

# 잘라낸 클립을 저장하는 함수
def save_flicker_clip(video_path, clip_ranges, save_path, width, height, fps):
    """
    깜빡임이 발생한 구간을 잘라서 새로운 비디오로 저장합니다.
    
    :param video_path: 원본 비디오 파일 경로
    :param clip_ranges: 시작과 끝 프레임을 포함한 튜플 리스트
    :param save_path: 클립을 저장할 경로
    :param width: 비디오 가로 크기
    :param height: 비디오 세로 크기
    :param fps: 초당 프레임 수 (비디오의 FPS)
    """
    # 결과 저장 디렉토리
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(video_path)
    for i, (start, end) in enumerate(clip_ranges):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        out_path = os.path.join(save_path, f"clip_{i+1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        for j in range(start, end+1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()

    cap.release()

# 메인 함수
def detect_pdp(video_path, save_path, threshold=0.2, area_ratio=0.1, min_flicker_hz=3, max_flicker_hz=50):
    """
    PDP 기준을 만족하는 깜빡임 구간을 검출하여 새로운 클립으로 저장합니다.
    
    :param video_path: 영상 파일 경로 (.mp4, .mov 등)
    :param save_path: 잘라낸 비디오 파일을 저장할 폴더 경로
    :param threshold: 휘도 차이 임계값 (상대 휘도 기준)
    :param area_ratio: 면적 비율 기준 (10% 이상이어야 함)
    :param min_flicker_hz: 최소 깜빡임 주파수 (초당 몇 번)
    :param max_flicker_hz: 최대 깜빡임 주파수 (초당 몇 번)
    """

    cap = cv2.VideoCapture(video_path) #영상 스트림 open
    fps = cap.get(cv2.CAP_PROP_FPS) #1초에 몇 프레임으로 구성되어 있는지 반환
    print(f"이 영상의 프레임 속도는 {fps} FPS입니다.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #전체 프레임 수수
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_area = width * height

    # 초기화
    prev_luminance = None
    flicker_frames = [] #가이드라인에 해당하는 프레임 번호를 담음
    result = [] #모든 프레임 결과를 담음

    # 저장 폴더가 없으면 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file = os.path.join(save_path, "flicker_results.csv")

    for idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        luminance = calculate_luminance(frame)

        if prev_luminance is not None:
            diff = calculate_luminance_diff(luminance, prev_luminance) #휘도 차이 계산
            flicker_detected = check_area_ratio(diff, prev_luminance, threshold, frame.shape[0] * frame.shape[1], area_ratio) #일정 비율을 넘는지 계산
            result.append((idx, flicker_detected))
            if check_area_ratio(diff, prev_luminance, threshold, total_area, area_ratio):
                flicker_frames.append(idx)

            frame_filename = os.path.join(save_path, f"frame_{idx}.jpg")
            cv2.imwrite(frame_filename, frame)

        prev_luminance = luminance

    cap.release()

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame Number', 'Flicker Detected'])  # 헤더 작성
        writer.writerows(result)  # 결과 데이터 작성

    print(f"결과가 {output_file}에 저장되었습니다.")
    print(f"모든 프레임 이미지가 {save_path}에 저장되었습니다.")

    # 깜빡임 구간 탐지 (그룹화)
    clip_ranges = detect_flickering(flicker_frames, fps, min_flicker_hz, max_flicker_hz)

    # 결과 클립 저장
    save_flicker_clip(video_path, clip_ranges, save_path, width, height, fps)

    print("Detected flickering segments:", clip_ranges)


def main():
    # 영상 파일 경로
    """
    video_path를 수정하게 되면 detect_pdp의 두 번째 매개변수도 변경
    """
    video_path = r"C:\kimjs\nova_guideline\meta\17_flickering_light.mp4"  # 경로 지정
    detect_pdp(video_path, "output_flickering_light")

if __name__ == "__main__":
    main()