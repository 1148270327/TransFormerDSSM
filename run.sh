#!/usr/bin/env bash
source activate HQpy36;
export PYTHONPATH=$PYTHONPATH:/home/yuanxianfeng/TransformerDSSM;
nohup python /home/yuanxianfeng/TransformerDSSM/Debug.py   >/home/yuanxianfeng/TransformerDSSM/DSSM_console.file  2>&1  &