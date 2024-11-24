# Hymba-1.5B-Instruct-Docker

## 概要
Hymba-1.5B-InstructおよびBaseのDocker版です。

## setup

```sh
ocker build -t hymba_env .
```

```sh
docker run --gpus all -it hymba_env
```

**Run Base Model**

```sh
python3 Hymba-1.5B-Base.py
```

**Run Instruct Model**

```sh
python3 Hymba-1.5B-Instruct.py
```
