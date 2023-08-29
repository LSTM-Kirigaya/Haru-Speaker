# Haru-Speaker



> 尚未完成，请勿下载，这个仓库只是为了方便进行同步。

## 前言

一个开箱即用的VITS中文语音合成项目，基于[vits-uma-genshin-honkai](https://github.com/wuheyi/vits-uma-genshin-honkai)项目开发，如果star本项目，请star `vits-uma-genshin-honkai`。


## 安装步骤

> 先自己去 [pytorch.org](https://pytorch.org) 装 torch !!!

```bash
# 安装python库
$ pip install -r requirements.txt

# Windows 下载模型
$ wget -O model/G_953000.pth https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/resolve/main/model/G_953000.pth

# Linux 下载模型
$ wget -P model/ https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/resolve/main/model/G_953000.pth
```


## 启动

```bash
# 使用 cpu
$ python app.py --device cpu

# 使用 gpu
$ python app.py --device cuda
```