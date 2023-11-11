# taxi-Q-learning
 Q-러닝 알고리즘을 사용하여 강화 학습을 구현 합니다.

import gym
import numpy as np
import cv2
from IPython.display import HTML
from pathlib import Path

# 환경 생성
env = gym.make('Taxi-v3')

# Q 테이블 초기화
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size, action_space_size))

# 하이퍼파라미터 설정
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 0.1
num_episodes = 1000

# 강화 학습 알고리즘 구현 (Q-Learning)
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # epsilon-greedy 정책으로 행동 선택
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()  # 랜덤 행동 선택
        else:
            action = np.argmax(q_table[state, :])  # 최적 행동 선택

        # 선택한 행동으로 환경 상태 업데이트
        next_state, reward, done, _ = env.step(action)

        # Q 값 업데이트
        q_value = q_table[state, action]
        max_next_q_value = np.max(q_table[next_state, :])
        new_q_value = (1 - learning_rate) * q_value + learning_rate * (reward + discount_factor * max_next_q_value)
        q_table[state, action] = new_q_value

        # 상태 업데이트
        state = next_state

# 최종적으로 학습된 Q 테이블 출력
print(q_table)

# 학습된 에이전트의 행동 시각화
state = env.reset()
frames = []

while True:
    action = np.argmax(q_table[state, :])
    next_state, reward, done, _ = env.step(action)
    frames.append(env.render(mode='rgb_array'))
    state = next_state
    if done:
        break

# 이미지로 변환하여 시각화
height, width, _ = frames[0].shape
video_path = 'taxi_agent.mp4'
video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

for frame in frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

video.release()

# 동영상 파일을 출력
video_path = Path(video_path).absolute()
HTML(f'<video width="400" height="300" controls><source src="{video_path}" type="video/mp4"></video>')


from google.colab import files

# 동영상 파일을 다운로드
files.download('taxi_agent.mp4')
