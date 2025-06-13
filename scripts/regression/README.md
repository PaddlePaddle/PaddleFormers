## PaddleNLP CI

主要代码结构及说明：
```
./
├── run_ci.sh # CI pr级别 执行入口
├── run_release.sh # 天级别回归&发版 执行入口
├── ci_case.sh # CI 核心case
├── ci_normal_case.py # 规范模型case 执行脚本
└── requirements_ci.txt # 依赖库
```
