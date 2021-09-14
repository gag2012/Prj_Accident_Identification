# YOLOv3을 이용한 영상처리 시스템 구현
운전자 및 동승자 영상 추출 skeleton 및 딥러닝을 이용한 교통사고 모니터링 시스템 개발 (딥러닝, YOLOv3)
  
- 프로젝트 설명 동영상  
[![Video Label](http://img.youtube.com/vi/D_dUjy80sQ4/0.jpg)](https://youtu.be/D_dUjy80sQ4?t=0s)     


## :computer: Used
- YOLOv3 
- Darknet
- Head Pose Estimation
- Google Colab
- Python
- opencv2, numpy
- slovePnP Algorithm

## :+1: Achievement
1) 졸업 논문 제출 
2) 한국통신학회 동계 학술대회 학부부문 참여
https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10547834
3) 한국통신학회 논문 투고 
4) 슈퍼챌린지 해커톤 참여 : 인기상 수상


## :memo: Summary
기존의 차량 긴급구난체계 서비스의 경우 사고 위치 및 발생 시각과 같은 사고에 대한 정보는 정확히 제공되지만, 환자의 상태에 대한 정보가 부족하여 정확한 응급조치를 하기 어렵다는 단점이 있다. 본 시스템은 교통사고 발생 시 블랙박스 내부 영상과 제안된 객체 인지 알고리즘을 구현한 응용시스템을 통해 사고상황을 비교적 정확히 감지하고, 실시간으로 탑승자의 부상과 관련된 정보를 제공하는 시스템을 제안한다. 본 시스템에서는 사고상황을 정확히 파악하기 위해 YOLO를 이용하여 사고의 형태를 구분한다. 사고로 판단될 경우, 머리 자세 추정 기법과 HSV(Hue Saturation Value) 컬러 모델을 이용하여 탑승자의 부상 정도와 출혈 여부 등의 상태를 파악해 관련 정보를 수집한 후, 탑승자의 휴대폰과 연동된 어플리케이션을 이용하여 긴급 구조기관에 전송한다.

**이용 데이터 수**  
학습 데이터와 테스트 데이터를 7:3 비율로 구성했으며 자세한 내역은 아래와 같다.

| 이용 데이터 갯수              | 이미지  | 동영상 |
| :--------------------------- |:------:| -----:|
| (Leanring) normal state      | 1000   | 62    |
| (Leanring) accident state    | 1000   | 55    |
| (Test) normal state          | 855    | 29    |
| (Test) accident state        | 50     | 15    |


## :mag: Content
- 제안된 긴급구난체계 서비스 시스템 모델  
![image](https://user-images.githubusercontent.com/40004210/133212792-a7d027bc-d1ae-4432-a32f-3a50a896dd5c.png)  
교통사고 발생 시, 학습된 딥러닝 모델이 사고를 감지한다. 이후 탑승자의 목꺾임 정도와 출혈량 계산을 진행하고 데이터를 가공한 후 응급기관에 전송한다.


- 교통사고 발생 시, 사고 감지  
![image](https://user-images.githubusercontent.com/40004210/133213065-76072736-dc40-45f6-93fd-f9aff8bfc55d.png)
![image](https://user-images.githubusercontent.com/40004210/133213077-31fb1970-57a0-4aa2-abd7-1191c6c0139e.png)  


- 교통사고 발생 시, 목 꺾임 감지   
![image](https://user-images.githubusercontent.com/40004210/133213285-3c0f0a3a-074d-4f60-95a2-d52e6c933edc.png)
![image](https://user-images.githubusercontent.com/40004210/133213290-6d9489e2-4a54-4070-967b-42ae3f08838b.png)  


- 긴급 구조기관에 전송하기 위해 데이터 수집 및 처리 시스템에 의해 처리된 데이터  
![image](https://user-images.githubusercontent.com/40004210/133213328-12210dac-530d-46d5-ad95-ea8a218fb5bc.png)  

## :clipboard: Result
**정확도 검증 결과**  
| Test classification                | Test item | Accuracy |
| :--------------------------------- |:---------:| --------:|
| drive accident and recognize face  | Drive     | 94%      |
| drive accident and recognize face  | Accident  | 96%      |
| angle of neck joint                | Pitch     | 94.9%    |
| angle of neck joint                | Roll      | 92.6%    |
| detect bleeding                    | Blood loss| 76%      |

**결론**  
 교통사고가 발생했을 때 짧은 시간 내에 정확한 응급 처치와 적절한 병원 선정을 위한 차량 긴급구난체계 시스템에 대해 제안하였다. 제안된 시스템은 사고 감지 여부 판단 시스템, 환자 상태 분석 시스템, 데이터 처리 및 전송 시스템으로 나뉘어 각각 사고가 발생했을 때 사고 감지 여부 판단과 환자 상태 분석을 진행한다.   
 
 먼저 사고 감지 여부를 확인하기 위해, 사고에 대해 학습된 YOLOv3 알고리즘을 사용하여 사고 여부를 판단한다. 또한, 사고로 인한 탑승자의 상태를 분석하기 위해 YOLOv3 알고리즘을 활용하여 얼굴에 대한 바운딩 박스와 특징점을 추출한다. 추출된 특징점들과 solvePnP 알고리즘을 사용하여 머리의 위치에 대해 추정하고, 사고 전후에 대한 환자의 목꺽임 정도를 분석하였다. 또한, HSV 컬러 모델을 사용하여 바운딩 박스 내부의 출혈량을 분석하였다. 마지막으로 해당 데이터들을 이용하여 환자에 상태를 수집하고, 사고 관련 정보와 함께 차량 긴급구난체계 서비스를 제공하는 기관에 전송 한다.  

 제안된 시스템은 사고상황에 대한 성능 평가를 통해 약 95%의 정확도로 사고 감지를 하고, 93% 이상의 정확도로 환자의 현재 증상을 분석하며, 76% 정도의 정확도로 출혈량을 판단할 수 있는 것을 확인하였다. 따라서 본 논문에서 제안된 시스템을 통해, 빠르게 사고 여부를 판단하고 환자의 상태를 분석하여 정확한 응급조치를 할 수 있도록 도와주는 차량 긴급구난체계 서비스를 제공할 수 있음을 보였다. 향후 해당 서비스를 통해 교통사고로 인한 사망률을 크게 줄일 수 있을 것으로 기대된다.
