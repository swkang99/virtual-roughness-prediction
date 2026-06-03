Project: virtual-tactile-property

Input : Texture PBR maps -> Output : Roughness

\[ 실행 순서 \]

git clone --recuesive https://github.com/swkang99/virtual-tactile-property.git

cd ./virtual-tactile-property/src

conda create -n virtual-tactile-property python=3.13

pip install -r requirements.txt 

python -m pip install opencv-python

필요 시 실행 => python3 ./data/split_dataset.py --data-dir ./data/MOESM --train-ratio 80 --valid-ratio 15 --test-ratio 5  

python3 train.py

python3 test.py
