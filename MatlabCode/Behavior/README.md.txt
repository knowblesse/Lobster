# Files
## functions
[**BehavDataParser**](#behavdataparser) : TDT Open Bridge 에서 추출한 행동 데이터 csv 파일을 불러온다.

[**createGaussianHitmap**](#creategaussianhitmap) : XY data로부터 hitmap을 그려냅니다.

[**drawmap**](#drawmap) : createGaussianHitmap 데이터를 그려주는 함수. (createGaussianHitmap 함수 내에서 dodraw를 true로 설정하면 필요 없음.)

[**firedLocationParser**](#firedlocationparser) : spike time 데이터가 있는 txt 파일을 읽어서 위치와 연결해준다.

[**getHeadDegree**](#getheaddegree) : 이전좌표와 이후 좌표를 기반으로 움직인 방향의 각도를 구합니다. (X+ 축이 0도, Y+축이 90도 입니다.)

[**locationParser**](#locationparser) : XYXY csv 파일을 읽어서 위치 값들을 뽑아 출력한다.(recoverLocData 포함)

[**recoverLocData**](#recoverlocdata) : tracking loss로 인한 데이터 손실 부분을 주변 데이터를 토대로 추측해서 재구성한다.

## Scripts
**AnalyticValueExtractor** : BehavDataParser를 돌린 후 돌리면 각 trial 별 Avoid(A) / Escape(E) / 1 min timeout(M) / Give up(G) / No Lick(N) 값을 출력한다.


### BehavDataParser
- input
	- **targetdir : string(path) :** csv 파일들이 있는 경로
- output
	- **ParsedData : {1,4} cell :** {[TRON Time, TROF Time], [[IRON, IROF]], [[LICK,LOFF]], [[ATTK, ATOF]]}
	- **Trials : [n, 2] array :** [TRON Time, TROF Time]
	- **IRs : [n, 2] array :** [IRON, IROF]
	- **Licks : [n,2] array :** [LICK,LOFF]
	- **Attacks : [n,2] array :** [ATTK, ATOF]

### createGaussianHitmap
- input
	- **xydata : [n, 2] array :** [xdata, ydata]
	- **distribution : double :** gaussian 분포를 만들기 위한 sigma 값.
	- **stride : int :** 처리 속도 향상을 위해서 넣은 숫자만큼 데이터를 건너뛰어 읽음. 1로 설정시 모든 데이터를 사용.
	- **axis_limit : [1, 4] array :** [xmin, xmax, ymin, ymax] hitmap 데이터의 x축의 범위, y축의 범위를 설정해줌.
	- **dodraw : boolean :** true로 설정시 hitmap 그래프를 그림. false로 설정시 나중에 따로 그려주어야 함.
- output
	- **map : [n, n] array :** hitmap data. surf 함수를 사용해서 그릴것.

### drawmap
- input
	- **map : [n, n] array :** hitmap data. createGaussianHitmap 함수에서 출력됨.
	- **axis_limit : [1, 4] array :** [xmin, xmax, ymin, ymax] hitmap 데이터의 x축의 범위, y축의 범위를 설정해줌.
	- **c_axis_limit : [1, 2] array :** colorbar를 그리기 위한 한계값. 
- output
	- **map : [n, n] array :** hitmap data. surf 함수를 사용해서 그릴것.

### firedLocationParser
- input
	- **result : [n, 6] array :** [시간, Red X, Red Y, Green X, Green Y, Head Direction] locationParser가 뱉어내는 result 파일. (실제로는 첫 Col 만 사용)
- output
	- **firedIndices : [n, 1] array :** locationParser의 result 파일을 기준으로 unit이 fire한 시간과 가장 가까운 indices

### getHeadDegree
- input
	- **x1 : double :** t 시간의 x좌표 
	- **y1 : double :** t 시간의 y좌표
	- **x2 : double :** t+1 시간의 x좌표
	- **y2 : double :** t+1 시간의 y좌표
- output
	- **head_degree : double :** 각도.

### locationParser
- input
	- **dodraw : boolean :** 결과값을 그래프로 그릴 지 말지를 결정한다.
- output
	- **result : [n, 6] array :** [시간, Red X, Red Y, Green X, Green Y, Head Direction] 
- warning
	- Col 2-5의 데이터는 open bridge 설정에 따라 Red와 Green이 바뀔 수 있으므로 주의할 것.

### recoverLocData
- input
	- **originalLocData : {2, 2} cell :** { Red X , Green X; Red Y, Green Y}
	- **X_RANGE : [1,2] array :** 정상적인 X 값의 범위. [xmin, xmax]
	- **Y_RANGE : [1,2] array :** 정상적인 Y 값의 범위. [ymin, ymax]
- output
	- **recoveredLocData : {2, 2} cell :** { Red X , Green X; Red Y, Green Y}
- warning
	- open bridge 설정에 따라서 정상적인 X, Y의 범위를 나타내는 X_Range, Y_Range 값을 바꿔줘야할 경우가 있음.