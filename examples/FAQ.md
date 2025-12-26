# 常见问题
## 1. 多卡并行训练时出现通信问题
### 问题描述
在使用 examples 中的多卡训练指令时，出现类似以下错误信息：
```
LAUNCH INFO 2025-10-29 19: 08: 08, 155 Waiting peer start..
```
长时间没有响应。

### 解决方案
在终端中输入以下指令
```
unset PADDLE_TRAINERS_NUM
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
```


## 2. 网络环境问题
### 问题描述
使用 huggingface 源下载模型或训练时出现类似以下错误信息：
```
[2025-10-30 20:28:41,961] [ WARNING] _util.py:319 - MaxRetriesError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /Qwen/Qwen3-0.6B-Base/resolve/main/tokenizer_config.json (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x[...HIDDEN_ADDRESS...]>), 'Connection to huggingface.co timed out. (connect timeout=10)')"), 'Request ID: xxxxxxx') thrown while requesting HEAD https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/tokenizer_config.json
```
### 解决方案
1. **（推荐）切换下载源查看是否能够解决问题。**
2. 将模型下载到本地后使用本地目录执行。
