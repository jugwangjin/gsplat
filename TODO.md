# Gaussian Splatting Simplification Implementation TODO

## 1. 렌더링 수식 정리
- 기본 렌더링 수식: $C = \sum^{N}_{i=1}{\alpha_{i}c_{i}T_{i}}$
- $T_{i}=\prod_{j}^{i-1}{(1-\alpha_{j})}$
- k번째 Gaussian 제거 시 렌더링: $\hat{C}_{k} = C - \alpha_{k}c_{k}T_{k} + \frac{1}{1-\alpha_{k}}(C-\sum^{k}_{i=1}{\alpha_{i}c_{i}T_{i}})$
- Loss 비교: $L = |GT-C|, \hat{L}_{k} = |GT-\hat{C}_{k}|$
- Score 계산: $S_{k}=\hat{L}_{k} - L$

## 2. 근사 가정
1. 한 Gaussian이 remove/prune 되어도 다른 Gaussian의 pixel rgb 값에 대한 기여도는 영향을 주지 않음
2. 업데이트할 대상 줄이기: 한 Gaussian이 remove/prune 될 때 최종 rendered rgb에만 영향
3. Gaussian들이 2D에 projection 된 영역은 대체로 비슷한 색을 가짐

## 3. 구현해야 할 파일들

### 3.1 gsplat/cuda/csrc/rasterize_to_pixels_fwd_approx.cu
- [ ] 각 Gaussian의 contribution 저장, 픽셀별 정보를 따로 저장할 수 없으므로 (메모리 제한), 누적하여 평균을 구할 예정
누적할 데이터
  - cur_color: 색 * alpha * transmission (cur_color)
  - final rendered color (pix_out)
  - Transmission so far (T)
 
Transmission을 저장할 필요가 있는지, 확인이 필요함. 

max weight and depth -> 이 것은 다른 스크립트를 위해 필요하므로 유지 
potential loss 구하는 파트 -> 필요없어짐짐


### 3.3 gsplat/cuda/csrc/bindings.cpp
- [ ] 새로운 텐서들을 Python 인터페이스에 추가
  - cur_color tensor
  - depth information tensor
  - visibility information tensor
- [ ] 새로운 함수들을 바인딩
  - potential loss 계산 함수
  - Gaussian pruning 함수

### 3.4 gsplat/cuda/csrc/types.cuh
- [ ] 새로운 데이터 구조 정의
  - Gaussian contribution 구조체
  - Depth/Visibility 정보 구조체

### 3.5 gsplat/cuda/csrc/bindings.h
- [ ] 새로운 함수 선언 추가
  - potential loss 계산 함수
  - Gaussian pruning 함수

## 4. 검증 방법

### 4.1 Depth - Potential Loss Plot
- [ ] 각 Training View마다 potential loss vs depth graph plot
- [ ] Depth 정규화 구현
  - 픽셀별 max depth로 나누기
- [ ] Potential loss 정규화 구현
- [ ] 상관관계 분석

### 4.2 Visibility - Potential Loss Plot
- [ ] Visibility vs potential loss graph plot
- [ ] 상관관계 분석

### 4.3 Pruning 검증
- [ ] Weight - potential loss plot
- [ ] Depth - potential loss plot
- [ ] Prune되는 Gaussian들의 특성 분석
  - Weight가 낮은지
  - Depth가 높은지

## 5. Approximated Potential Loss Update 구현
- [ ] 각 view마다 final rendered color, cur_color 저장
- [ ] Pruning 시 loss 차이 계산
  - gt - final rendered color vs gt - (final rendered color - cur_color)
- [ ] Camera space에서의 2D distance 기반 계산
- [ ] Final rendered color 업데이트
  - final_rendered_color - cur_color
- [ ] 정확도 향상을 위한 추가 계산
  - accumulated color + (final rendered color - accumulated color - cur_color) / (1 - alpha)

## 6. 성능 최적화
- [ ] 메모리 사용량 최적화
- [ ] 계산 속도 최적화
- [ ] 렌더링 재시작 시점 최적화 