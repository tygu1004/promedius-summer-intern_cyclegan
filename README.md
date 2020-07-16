# Cycle-gan 코드를 tensorflow 2.x 버전으로 수정하였습니다.

기존 코드 : https://github.com/hyeongyuy/CT-CYCLE_IDNETITY_GAN_tensorflow

# 코드 단위에서 수정한 사항

  1. data input pipeline을 queue에서 tf.data로 바꾸었습니다.
  2. module을 tf.keras model을 이용하도록 수정하였습니다.
  3. 학습 과정을 @tf.function을 이용하도록 수정하였습니다.
  4. summary 과정을 session을 이용하지 않는 방식으로 수정하였습니다.
  5. DCMDataloader class의 크기를 줄였습니다.
  6. 기존의 코드에 있던 cyclegan model에 해당하지 않는 코드를 삭제하였습니다.
  7. 기존의 코드에 있던 unpaired data를 다루는 것에 해당하지 않는 코드를 삭제하였습니다.
  
# 학습 과정에서 변경된 부분

  1. learning rate decay 방식 변경
  
    기존의 코드는 "end_epoch"이라는 argument를 입력받아서 학습 과정이
    "decay_epoch" 이상 진행 중일 때 learning rate를 decay 하는 방식이었습니다.
    하지만 기존에 사용하는 Adam optimaizer에 decay되는 성질이 있기 때문에
    위의 방식이 불필요하다고 생각했습니다.
    그리고 위의 방식은 다른 시점에서 checkpoint를 이용하여 학습을 이어갈 때
    완벽하게 동일한 상황을 재현하기가 어려웠습니다.
    그래서 end_epoch과 decay_epoch을 이용하여 learning rate를
    decay하는 방식을 포기하고 Adam의 성질 그대로를 이용하도록 변경하였습니다.
    
  2. 학습 진행을 epoch에서 step단위로 변경
  
    step 단위로 전체 학습과정을 생각하도록 바꾸었고 "epoch" argument를 추가하여
    주어진 데이터를 몇 번 만큼 순회할 지 선택하였습니다.
    기존 코드는 전체 데이터 크기가 너무 큰 관계로 한 epoch을 수행하는 것이 전체 데이터가 아닌
    random하게 sample된 데이터를 "step_per_epoch"만큼만 학습하는 것이었습니다.
    하지만 전체 데이터를 학습하는 것이 좋겠다고 생각했고 step 단위로 학습과정을 생각하게
    되었으므로 "step_per_epoch"을 삭제하고 epoch을 전체 데이터를 순회하는 단위로 바꿨습니다.
    
# main.py argument 변경사항
  
  - 삭제된 항목
  
    '--model'
    
    '--end_epoch'
    
    '--decay_epoch'
    
    '--unpair'
    
    '--dcm_path'
  
  - 추가된 항목
  
    '--epoch' : 학습하고자 하는 epoch 수
    
    '--data_path' : 데이터가 저장된 디렉토리
 
    '--expansion' : 데이터 파일 확장자 (dcm or pbg)
