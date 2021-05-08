conda create -n hico_env python=3.7
conda activate hico_env

git clone git@github.com:NguyenVanThanhHust/detectron2.git

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

cd detectron2
python -m pip install -e .
