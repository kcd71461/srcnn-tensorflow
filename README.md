# srcnn-tensorflow

### directory structure
```
./datasets: training bitmap files
./checkpoint: checkpoint save dir. model will saved in "./checkpoint/srcnn".
./logs: summary logs
./test: test images dir
./result: test ouput dir
```


### script
```
generate_train_h5: generate "train.h5" file
train.py: train
test.py: generate original, bicubic, srcnn results
```


### How to use
#### learning
1. ./datasets 경로에 학습할 이미지(.bmp)를 넣고
2. "generate_train_h5.py"를 실행시켜 train.h5파일을 생성하고
3. "train.py" 실행


#### test
1. ./test 경로에 테스트할 이미지(.bmp)를 넣고
2. "test.py --test_img {파일명}" 실행. (./test 경로내의 이미지 파일명)
