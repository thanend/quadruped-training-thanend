# 01 MuJoCo 环境搭建与 URDF 模型导入

## 1. 任务完成内容
- 完成 MuJoCo 3.14.0 环境部署与基础使用
- 将 N-W-wolf 机器狗 URDF 转换为 MJCF 格式并导入仿真
- 添加平坦地面场景，配置关节力矩驱动器
- 实现机器狗在地面静止趴下的效果，关节输出力矩为0

## 2. 环境与依赖
- 系统：Linux (Ubuntu)
- 仿真引擎：MuJoCo 3.14.0
- 编程语言：Python 3
- 依赖库：mujoco-python

## 3. 运行方式
1. 确保已安装 MuJoCo 与 Python 环境
2. 终端进入当前目录
3. 执行命令：
```bash
python3 run_sim.py
