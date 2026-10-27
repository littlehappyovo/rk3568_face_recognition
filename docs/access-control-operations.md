# 局域网门禁运行与恢复

本说明对应此源码版本。当前方案为板端单路本地识别、电脑浏览器远程管理；硬件部署须先完成候选构建和备份，验收状态见[验证记录](access-control-validation.md)。

## 服务与目录

Qt主程序启动摄像头、识别后端和Python子进程。Python在项目独立venv运行Flask/Waitress、SQLite和认证；不需要新增root服务。实际部署时按设备环境配置已有服务及GDM启停权限，不扩大sudo范围。

默认运行目录为`~/.local/share/PortableFaceSystem/FaceRecognitionSystem`；`FACEAPP_DATA_DIR`可覆盖。原`data/`、图库、音量偏好继续使用，新文件位于`access/`：

- `access.sqlite3`及可能存在的`-wal`、`-shm`：事件、配对、异常处理历史、终端配置、密码派生值和会话。
- `outbox/`：Qt尚未收到数据库持久化确认的事件。收到同UUID确认后才删除；重试不会增加记录。
- `control.sock`、`runtime.lock`：仅本用户可访问的进程间通道和运行锁。服务随机密钥只经标准输入传给子进程。

`access/`权限0700。不要公开运行目录、真实照片或数据库。HTTP地址只提供管理业务接口，不提供SQL或任意文件读取。

## 安装和启动

先准备模型与RKNN SDK，安装项目后台环境：

```bash
bash scripts/setup_access_server.sh
```

默认venv为`server/.venv`。使用其他目录可在安装时设置`FACEAPP_ACCESS_VENV`，并在启动环境设置对应`FACEAPP_ACCESS_PYTHON`。依赖从`server/requirements.lock`安装，板端Python须为3.10或更新兼容版本。

部分板卡Ubuntu镜像没有`ensurepip`，`python3 -m venv`会提示缺少`python3.10-venv`。无需改系统Python：可用`python3 -I -m venv --without-pip server/.venv`建立项目环境，按[PyPA官方安装说明](https://pip.pypa.io/en/stable/installation/)下载`get-pip.py`并在此venv内执行，再用`server/.venv/bin/python -I -m pip --require-virtualenv install --only-binary=:all: -r server/requirements.lock`安装锁定依赖。不要在系统Python运行引导或安装命令。此路径曾在ARM64板卡验证，Pillow和MarkupSafe使用ARM64 wheel。

```bash
FACEAPP_BUILD_DIR="$PWD/build-access-candidate" \
FACEAPP_VERIFY_BUILD_DIR="$PWD/build-access-verification" \
FACEAPP_JOBS=2 bash scripts/build_rk3568.sh
```

使用新目录避免覆盖正式程序。构建脚本保留已有目录的CMake生成器；新目录使用CMake默认生成器。服务测试的Python独立使用`FACEAPP_ACCESS_PYTHON`。

管理服务默认在所有网络接口监听8080端口。端口冲突时设置`FACEAPP_ACCESS_PORT`；不要终止无关服务。`FACEAPP_ACCESS_PYTHON`和端口需设置在实际服务启动环境，交互SSH的export不会自动进入已有systemd服务。

接口目前使用明文HTTP，密码、会话和人脸资料在网络传输中未加密；仅在可信隔离局域网使用，对外或生产部署应另行配置HTTPS及网络访问控制。

后台未就绪时摄像头继续运行，界面提示管理故障，记录保存在待发队列；管理登录不可用。恢复后台后有序补发。磁盘写入失败会提示故障，不能当作已记录。

## 首次管理

板卡启动后默认仅显示预览与识别结果。点击“管理”，使用实体键盘设置8至128字符的独立密码，再进入原管理界面。密码框自动获得焦点，Enter提交、Esc取消；当前版本不提供屏幕软键盘。不要将密码写入源码、命令参数或日志。

电脑浏览器访问`http://板卡IP:8080`并登录，可查看记录/配对/异常、管理人员与照片、重建图库及切换板端站点方向。电脑通过网页管理，可上传人员样本照片，但不采集或上传实时视频供识别。板端和网页闲置5分钟退出；自动画面和轮询不会延长会话。修改管理密码会撤销已有会话。

板端站点默认进，管理员可将其切换为出，以同一台摄像头分时记录方向。方向取服务端版本，历史方向不改。重复进入或无进入却离开立即标记；未离开默认24小时后标记，可按分钟调整或设0关闭。超时后离开会闭合配对，同时保留历史异常及处理备注。升级时会撤销旧电脑终端令牌并禁止新电脑端事件写入；已有电脑终端记录保留查询、导出和历史配对。

启禁用、删样本等变化可能使图库失效，重建完成前不会继续使用旧身份特征。批量导入照片逐张串行处理，失败时已完成部分仍存在，应刷新核对后继续。

## 备份与替换

以下步骤仅在候选版本本机/ARM64/RKNN测试通过且板卡可联调时执行。先核对当前源码、二进制、服务和运行目录，不把历史路径或状态当成现状。

1. 保持正式服务运行，在新构建目录完成候选测试。
2. 停止现有人脸服务并确认Qt和Python子进程已退出。
3. 在私有备份目录保存旧二进制、配置、变更源码和完整运行目录，记录原始人脸资料/样本/图库的哈希。完整运行目录包含SQLite及待发队列。
4. 停写后可完整复制数据库与其WAL/SHM；若需对活动数据库取一致快照，使用Python`sqlite3.Connection.backup()`，不能只复制主文件。待发队列仍需在停写后备份，不能假定数据库快照包含队列。
5. 替换候选程序和所需服务文件，使用既有权限启动正式服务。管理密码在板卡设置。
6. 核对值守与管理界面、默认相机/识别、网页管理、板端方向切换、历史记录和原语音。

## 回退

失败时停止新程序，保留新`access/`和待发队列的独立副本，再恢复旧二进制及其配置、服务资源。不要用旧快照覆盖已经新增的事件数据库或真实人脸资料。旧程序不消费新记录数据库；后续恢复新程序时仍需使用保留的新数据库和待发队列。

回退后重新核验服务、相机、图库与语音；记录回退发生的版本和数据边界。本文档不表示这些板端步骤已经执行。
