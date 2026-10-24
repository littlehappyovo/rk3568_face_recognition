#include "mainwindow.h"

#include "facebackendservice.h"
#include "frameview.h"
#include "facedatabasedialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTimer>
#include <QVBoxLayout>

// MainWindow 只负责“控件是什么、摆在哪里、事件发给谁”。摄像头和模型不在
// 这里直接执行，而是通过 FaceBackendService 的 signal/slot 交给后台对象。
MainWindow::MainWindow(FaceBackendService *backend, QWidget *parent)
    : MainWindow(backend, true, parent)
{
}

MainWindow::MainWindow(FaceBackendService *backend, bool autoStart, QWidget *parent)
    : QMainWindow(parent)
    , m_backend(backend)
{
    buildUi();
    connectBackend(backend);
    updateControls();

    // 延迟到当前窗口构造完成后再发初始化请求，避免 backend 收到请求时控件
    // 还没有创建完。singleShot(0) 不是等待很久，而是放到下一轮事件循环。
    QTimer::singleShot(0, this, [this, autoStart] {
        emit configureBackendRequested(m_backendCombo->currentData().toString());
        emit refreshCamerasRequested();
        if (autoStart) emit autoStartRequested();
    });
}

void MainWindow::buildUi()
{
    // buildUi() 只创建控件和布局；真正的业务动作由下面 connectBackend() 建立的
    // signal/slot 连接完成。阅读本函数时可以先把它当成一张界面结构图。
    setWindowTitle(tr("人脸检测与识别系统"));
    resize(1440, 850);

    auto *central = new QWidget(this);
    auto *rootLayout = new QVBoxLayout(central);
    rootLayout->setContentsMargins(12, 12, 12, 8);
    rootLayout->setSpacing(10);

    auto *sourceGroup = new QGroupBox(tr("输入与后端"), central);
    auto *sourceLayout = new QHBoxLayout(sourceGroup);
    m_backendCombo = new QComboBox(sourceGroup);
    m_backendCombo->addItem(tr("OpenCV / LBPH（便携基线）"), QStringLiteral("opencv_lbph"));
    m_backendCombo->addItem(tr("ONNX 人脸特征（扩展）"), QStringLiteral("onnx_embedding"));
    m_backendCombo->addItem(tr("RKNN / MobileFaceNet（ArcFace）"),
                             QStringLiteral("mobilefacenet_rknn"));
    m_backendCombo->setCurrentIndex(m_backendCombo->findData(
        QStringLiteral("mobilefacenet_rknn")));
    m_cameraCombo = new QComboBox(sourceGroup);
    m_cameraCombo->setMinimumWidth(230);
    m_refreshButton = new QPushButton(tr("刷新设备"), sourceGroup);
    m_startButton = new QPushButton(tr("启动摄像头"), sourceGroup);
    m_stopButton = new QPushButton(tr("停止"), sourceGroup);
    m_importButton = new QPushButton(tr("导入图片…"), sourceGroup);

    sourceLayout->addWidget(new QLabel(tr("识别后端："), sourceGroup));
    sourceLayout->addWidget(m_backendCombo);
    sourceLayout->addSpacing(12);
    sourceLayout->addWidget(new QLabel(tr("视频设备："), sourceGroup));
    sourceLayout->addWidget(m_cameraCombo, 1);
    sourceLayout->addWidget(m_refreshButton);
    sourceLayout->addWidget(m_startButton);
    sourceLayout->addWidget(m_stopButton);
    sourceLayout->addWidget(m_importButton);
    rootLayout->addWidget(sourceGroup);

    auto *bodySplitter = new QSplitter(Qt::Horizontal, central);
    bodySplitter->setChildrenCollapsible(false);

    auto *viewsWidget = new QWidget(bodySplitter);
    auto *viewsLayout = new QVBoxLayout(viewsWidget);
    viewsLayout->setContentsMargins(0, 0, 0, 0);
    auto *framesSplitter = new QSplitter(Qt::Horizontal, viewsWidget);
    framesSplitter->setChildrenCollapsible(false);

    auto makeFramePanel = [framesSplitter](const QString &title, FrameView **view) {
        auto *panel = new QWidget(framesSplitter);
        auto *layout = new QVBoxLayout(panel);
        layout->setContentsMargins(0, 0, 0, 0);
        auto *titleLabel = new QLabel(title, panel);
        titleLabel->setAlignment(Qt::AlignCenter);
        titleLabel->setStyleSheet(QStringLiteral("font-weight: 600; padding: 5px;"));
        *view = new FrameView(panel);
        layout->addWidget(titleLabel);
        layout->addWidget(*view, 1);
        return panel;
    };
    framesSplitter->addWidget(makeFramePanel(tr("原始画面"), &m_originalView));
    framesSplitter->addWidget(makeFramePanel(tr("已登记人员照片"), &m_recognizedView));
    m_recognizedView->setPlaceholder(tr("等待识别"));
    framesSplitter->setStretchFactor(0, 1);
    framesSplitter->setStretchFactor(1, 1);
    viewsLayout->addWidget(framesSplitter);
    bodySplitter->addWidget(viewsWidget);

    auto *controlsScroll = new QScrollArea(bodySplitter);
    controlsScroll->setWidgetResizable(true);
    // 控制栏要放得下"标签 + 输入框"而不出横向滚动条。板端 eglfs 的 DPI 很高，
    // 控件尺寸偏大，所以按像素给足宽度（画面区还剩 1200+，够两路显示）。
    controlsScroll->setMinimumWidth(620);
    controlsScroll->setMaximumWidth(760);
    auto *controls = new QWidget(controlsScroll);
    auto *controlsLayout = new QVBoxLayout(controls);

    auto *personGroup = new QGroupBox(tr("人员与样本采集"), controls);
    auto *personLayout = new QFormLayout(personGroup);
    m_personIdEdit = new QLineEdit(personGroup);
    m_personIdEdit->setObjectName(QStringLiteral("capturePersonId"));
    m_personIdEdit->setPlaceholderText(tr("如 IN_001"));
    m_personIdEdit->setValidator(new QRegularExpressionValidator(
        QRegularExpression(QStringLiteral("[A-Za-z0-9_-]{1,32}")), m_personIdEdit));
    m_personNameEdit = new QLineEdit(personGroup);
    m_personNameEdit->setObjectName(QStringLiteral("capturePersonName"));
    m_personNameEdit->setPlaceholderText(tr("显示姓名"));
    m_groupCombo = new QComboBox(personGroup);
    m_groupCombo->addItem(tr("组内"), QStringLiteral("in"));
    m_groupCombo->addItem(tr("组外"), QStringLiteral("out"));
    m_sampleCountSpin = new QSpinBox(personGroup);
    m_sampleCountSpin->setRange(10, 500);
    m_sampleCountSpin->setValue(10);
    m_sampleCountSpin->setSuffix(tr(" 张"));
    personLayout->addRow(tr("人员 ID："), m_personIdEdit);
    personLayout->addRow(tr("姓名："), m_personNameEdit);
    personLayout->addRow(tr("组别："), m_groupCombo);
    personLayout->addRow(tr("目标样本："), m_sampleCountSpin);

    auto *captureButtons = new QWidget(personGroup);
    auto *captureButtonsLayout = new QHBoxLayout(captureButtons);
    captureButtonsLayout->setContentsMargins(0, 0, 0, 0);
    m_captureButton = new QPushButton(tr("开始采集"), captureButtons);
    m_endCaptureButton = new QPushButton(tr("结束采集"), captureButtons);
    captureButtonsLayout->addWidget(m_captureButton);
    captureButtonsLayout->addWidget(m_endCaptureButton);
    personLayout->addRow(captureButtons);

    m_captureProgress = new QProgressBar(personGroup);
    m_captureProgress->setRange(0, 100);
    m_captureProgress->setValue(0);
    m_captureProgress->setFormat(tr("采集：%v / %m"));
    personLayout->addRow(m_captureProgress);

    m_rawCaptureButton = new QPushButton(tr("采集原始图像（20张）"), personGroup);
    personLayout->addRow(m_rawCaptureButton);
    m_rawCaptureProgress = new QProgressBar(personGroup);
    m_rawCaptureProgress->setRange(0, 20);
    m_rawCaptureProgress->setValue(0);
    m_rawCaptureProgress->setFormat(tr("原始图像：%v / %m"));
    personLayout->addRow(m_rawCaptureProgress);
    controlsLayout->addWidget(personGroup);

    auto *modelGroup = new QGroupBox(tr("模型与识别"), controls);
    auto *modelLayout = new QVBoxLayout(modelGroup);
    m_trainButton = new QPushButton(tr("训练 / 重建模型"), modelGroup);
    m_databaseButton = new QPushButton(tr("人脸数据库管理…"), modelGroup);
    m_databaseButton->setObjectName(QStringLiteral("faceDatabaseButton"));
    m_recognitionCheck = new QCheckBox(tr("开启实时人脸识别"), modelGroup);
    m_trainingProgress = new QProgressBar(modelGroup);
    m_trainingProgress->setRange(0, 100);
    m_trainingProgress->setValue(0);
    m_trainingProgress->setFormat(tr("训练：%p%"));
    m_modelLabel = new QLabel(tr("模型：尚未加载"), modelGroup);
    m_modelLabel->setWordWrap(true);
    modelLayout->addWidget(m_trainButton);
    modelLayout->addWidget(m_databaseButton);
    modelLayout->addWidget(m_recognitionCheck);
    modelLayout->addWidget(m_trainingProgress);
    modelLayout->addWidget(m_modelLabel);
    controlsLayout->addWidget(modelGroup);

    auto *runtimeGroup = new QGroupBox(tr("运行状态"), controls);
    auto *runtimeLayout = new QFormLayout(runtimeGroup);
    m_statusLabel = new QLabel(tr("空闲"), runtimeGroup);
    m_statusLabel->setWordWrap(true);
    m_fpsLabel = new QLabel(QStringLiteral("0.0"), runtimeGroup);
    m_fpsLabel->setWordWrap(false);
    m_fpsLabel->setMinimumWidth(560);
    m_fpsLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_targetCountLabel = new QLabel(QStringLiteral("0"), runtimeGroup);
    runtimeLayout->addRow(tr("状态："), m_statusLabel);
    runtimeLayout->addRow(tr("FPS："), m_fpsLabel);
    runtimeLayout->addRow(tr("目标数："), m_targetCountLabel);
    controlsLayout->addWidget(runtimeGroup);
    controlsLayout->addStretch(1);

    controlsScroll->setWidget(controls);
    bodySplitter->addWidget(controlsScroll);
    bodySplitter->setStretchFactor(0, 1);
    bodySplitter->setStretchFactor(1, 0);
    rootLayout->addWidget(bodySplitter, 1);

    // 分组框不许被拉高：eglfs 高 DPI 下表单控件偏大，一旦分组框吃掉多余高度，
    // 表单行就被拉长，下面的按钮会被顶出屏幕（用户看到的是"按钮点不动"）。
    for (QGroupBox *group : central->findChildren<QGroupBox *>()) {
        group->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    }

    // 板端 eglfs 按面板物理尺寸算出 471 DPI，样式里跟 DPI 挂钩的布局间距/边距
    // 也跟着放大：表单行距涨到 44px、分组框内边距涨到 69px，右栏整列于是超出
    // 1080 高，最下面的"训练/开启识别"跑出屏幕。这里把间距和边距夹回正常像素值。
    for (QLayout *layout : central->findChildren<QLayout *>()) {
        if (layout == nullptr) {
            continue;
        }
        if (layout->spacing() > 8) {
            layout->setSpacing(8);
        }
        const QMargins margins = layout->contentsMargins();
        int top = qMin(margins.top(), 10);
        if (qobject_cast<QGroupBox *>(layout->parentWidget()) != nullptr) {
            top = qMax(top, 28);  // 给分组框标题留位置，别和第一行压在一起
        }
        if (margins.top() > top || margins.bottom() > 10 || margins.left() > 10 ||
            margins.right() > 10) {
            layout->setContentsMargins(qMin(margins.left(), 10), top,
                                       qMin(margins.right(), 10),
                                       qMin(margins.bottom(), 10));
        }
    }

    setCentralWidget(central);

    statusBar()->showMessage(tr("准备就绪。采集多人画面时，请在右侧画面点击需要采集的目标。"));

    connect(m_backendCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this] {
        emit configureBackendRequested(m_backendCombo->currentData().toString());
    });
    connect(m_refreshButton, &QPushButton::clicked, this, &MainWindow::refreshCamerasRequested);
    connect(m_startButton, &QPushButton::clicked, this, &MainWindow::startSelectedCamera);
    connect(m_stopButton, &QPushButton::clicked, this, &MainWindow::closeInputRequested);
    connect(m_importButton, &QPushButton::clicked, this, &MainWindow::chooseImages);
    connect(m_captureButton, &QPushButton::clicked, this, &MainWindow::beginCapture);
    connect(m_endCaptureButton, &QPushButton::clicked, this, &MainWindow::endCaptureRequested);
    connect(m_rawCaptureButton, &QPushButton::clicked,
            this, &MainWindow::beginRawFrameCaptureRequested);
    connect(m_trainButton, &QPushButton::clicked, this, &MainWindow::trainModelRequested);
    auto *database = new FaceDatabaseDialog(m_backend, this);
    connect(m_databaseButton, &QPushButton::clicked, database, &QDialog::open);
    connect(database, &FaceDatabaseDialog::capturePersonSelected, this,
            [this](const PersonProfile &person) {
                m_personIdEdit->setText(person.personId);
                m_personNameEdit->setText(person.displayName);
                m_groupCombo->setCurrentIndex(m_groupCombo->findData(person.group));
            });
    connect(m_recognitionCheck, &QCheckBox::toggled, this, &MainWindow::recognitionRequested);
    connect(m_originalView, &FrameView::faceClicked, this, [this](qint64 trackId) {
        if (m_capturing) {
            emit selectCaptureTrackRequested(trackId);
            statusBar()->showMessage(tr("已选择采集目标 #%1").arg(trackId), 4000);
        }
    });
}

void MainWindow::connectBackend(FaceBackendService *backend)
{
    // 这些连接把“主线程的用户操作”和“后台线程的业务操作”串起来。
    // QueuedConnection 会把调用排进 backend 所在线程的事件队列，避免跨线程
    // 直接执行成员函数。
    Q_ASSERT(backend);
    connect(this, &MainWindow::autoStartRequested,
            backend, &FaceBackendService::startAutomatically, Qt::QueuedConnection);
    connect(this, &MainWindow::configureBackendRequested,
            backend, &FaceBackendService::configureBackend, Qt::QueuedConnection);
    connect(this, &MainWindow::refreshCamerasRequested,
            backend, &FaceBackendService::refreshCameras, Qt::QueuedConnection);
    connect(this, &MainWindow::openCameraRequested,
            backend, &FaceBackendService::openCamera, Qt::QueuedConnection);
    connect(this, &MainWindow::closeInputRequested,
            backend, &FaceBackendService::closeInput, Qt::QueuedConnection);
    connect(this, &MainWindow::importImagesRequested,
            backend, &FaceBackendService::importImages, Qt::QueuedConnection);
    connect(this, &MainWindow::beginCaptureRequested,
            backend, &FaceBackendService::beginCapture, Qt::QueuedConnection);
    connect(this, &MainWindow::selectCaptureTrackRequested,
            backend, &FaceBackendService::selectCaptureTrack, Qt::QueuedConnection);
    connect(this, &MainWindow::endCaptureRequested,
            backend, &FaceBackendService::endCapture, Qt::QueuedConnection);
    connect(this, &MainWindow::beginRawFrameCaptureRequested,
            backend, &FaceBackendService::beginRawFrameCapture, Qt::QueuedConnection);
    connect(this, &MainWindow::trainModelRequested,
            backend, &FaceBackendService::trainModel, Qt::QueuedConnection);
    connect(this, &MainWindow::recognitionRequested,
            backend, &FaceBackendService::setRecognitionEnabled, Qt::QueuedConnection);

    connect(backend, &FaceBackendService::camerasChanged,
            this, &MainWindow::onCamerasChanged);
    connect(backend, &FaceBackendService::frameReady,
            this, &MainWindow::onFrameReady);
    connect(backend, &FaceBackendService::stateChanged,
            this, &MainWindow::onStateChanged);
    connect(backend, &FaceBackendService::captureProgress,
            this, &MainWindow::onCaptureProgress);
    connect(backend, &FaceBackendService::rawCaptureProgress,
            this, &MainWindow::onRawCaptureProgress);
    connect(backend, &FaceBackendService::trainingProgress,
            this, &MainWindow::onTrainingProgress);
    connect(backend, &FaceBackendService::modelChanged, this,
            [this](const QString &version, const QString &summary) {
                m_modelLabel->setText(tr("模型：%1\n%2").arg(version, summary));
            });
    connect(backend, &FaceBackendService::recognitionEnabledChanged, this,
            [this](bool enabled) {
                const QSignalBlocker blocker(m_recognitionCheck);
                m_recognitionCheck->setChecked(enabled);
            });
    connect(backend, &FaceBackendService::errorOccurred,
            this, &MainWindow::onError);
}

void MainWindow::chooseImages()
{
    const QString startDirectory = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    const QStringList paths = QFileDialog::getOpenFileNames(
        this, tr("选择人脸图片"), startDirectory,
        tr("图片 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)"));
    if (!paths.isEmpty()) {
        PersonProfile person;
        person.personId = m_personIdEdit->text().trimmed();
        person.displayName = m_personNameEdit->text().trimmed();
        person.group = m_groupCombo->currentData().toString();
        bool saveToDataset = false;
        if (!person.personId.isEmpty() && !person.displayName.isEmpty()) {
            saveToDataset = QMessageBox::question(
                                this, tr("保存提取的人脸"),
                                tr("是否把每张单人脸图片提取后保存到 %1 [%2] 的数据集？\n"
                                   "包含 0 张或多张人脸的图片会安全跳过。")
                                    .arg(person.displayName, person.personId),
                                QMessageBox::Yes | QMessageBox::No,
                                QMessageBox::Yes) == QMessageBox::Yes;
        }
        emit importImagesRequested(paths, person, saveToDataset);
    }
}

void MainWindow::startSelectedCamera()
{
    const QString cameraId = m_cameraCombo->currentData().toString();
    if (cameraId.isEmpty()) {
        QMessageBox::warning(this, tr("未选择摄像头"), tr("请刷新并选择一个视频设备。"));
        return;
    }
    emit openCameraRequested(cameraId);
}

void MainWindow::beginCapture()
{
    PersonProfile person;
    person.personId = m_personIdEdit->text().trimmed();
    person.displayName = m_personNameEdit->text().trimmed();
    person.group = m_groupCombo->currentData().toString();
    if (person.personId.isEmpty() || person.displayName.isEmpty()) {
        QMessageBox::warning(this, tr("人员信息不完整"), tr("请填写人员 ID 和姓名。"));
        return;
    }
    emit beginCaptureRequested(person, m_sampleCountSpin->value());
}

void MainWindow::onCamerasChanged(const CameraList &cameras)
{
    const QString previousId = m_cameraCombo->currentData().toString();
    m_cameraCombo->clear();
    for (const CameraDescriptor &camera : cameras) {
        m_cameraCombo->addItem(camera.displayName, camera.id);
    }
    const int previousIndex = m_cameraCombo->findData(previousId);
    if (previousIndex >= 0) {
        m_cameraCombo->setCurrentIndex(previousIndex);
    }
    updateControls();
}

void MainWindow::onFrameReady(const FrameResult &frame)
{
    // frameReady 是后台线程发来的结构化结果。这里不做检测/识别，只把实时帧和
    // 人脸观察结果交给左侧 FrameView；右侧只有 referenceRevision 变化时才更新，
    // 避免每帧复制静态人员照片。
    m_originalView->setFrame(frame.originalImage, frame.faces, true);
    if (frame.referenceRevision != m_referenceRevision) {
        m_recognizedView->setReferenceCards(frame.referenceCards);
        m_referenceRevision = frame.referenceRevision;
    }
    const auto formatPercent = [](double value) {
        return value >= 0.0
            ? QString::number(value, 'f', 0) + QStringLiteral("%")
            : QStringLiteral("--");
    };
    const auto formatTemperature = [](double value) {
        return value >= 0.0
            ? QString::number(value, 'f', 1) + QStringLiteral("°C")
            : QStringLiteral("--");
    };
    m_fpsLabel->setText(
        tr("FPS %1 | CPU %2 | NPU %3 | 温度 %4 | 内存 %5")
            .arg(QString::number(frame.fps, 'f', 1))
            .arg(formatPercent(frame.systemMetrics.cpuUsagePercent))
            .arg(formatPercent(frame.systemMetrics.npuUsagePercent))
            .arg(formatTemperature(frame.systemMetrics.temperatureCelsius))
            .arg(formatPercent(frame.systemMetrics.memoryUsagePercent)));
    m_targetCountLabel->setText(QString::number(frame.faces.size()));
}

void MainWindow::onStateChanged(ServiceState state, const QString &message)
{
    m_state = state;
    if (state == ServiceState::Streaming || state == ServiceState::Capturing) {
        m_streaming = true;
    } else if (state == ServiceState::Idle || state == ServiceState::Error) {
        m_streaming = false;
    }
    m_capturing = state == ServiceState::Capturing;
    m_training = state == ServiceState::Training;
    m_statusLabel->setText(tr("%1：%2").arg(stateName(state), message));
    statusBar()->showMessage(message, 7000);
    updateControls();
}

void MainWindow::onCaptureProgress(int acceptedSamples, int targetSamples)
{
    if (targetSamples <= 0) {
        m_captureProgress->setRange(0, 100);
        m_captureProgress->setValue(0);
        m_captureProgress->setFormat(tr("采集：0 / 0"));
        return;
    }
    m_captureProgress->setRange(0, targetSamples);
    m_captureProgress->setValue(acceptedSamples);
    m_captureProgress->setFormat(tr("采集：%v / %m"));
}

void MainWindow::onRawCaptureProgress(int capturedFrames, int targetFrames)
{
    if (targetFrames <= 0) {
        m_rawCapturing = false;
        m_rawCaptureProgress->setRange(0, 20);
        m_rawCaptureProgress->setValue(0);
        m_rawCaptureProgress->setFormat(tr("原始图像：0 / 20"));
        updateControls();
        return;
    }
    m_rawCapturing = capturedFrames < targetFrames;
    m_rawCaptureProgress->setRange(0, targetFrames);
    m_rawCaptureProgress->setValue(capturedFrames);
    m_rawCaptureProgress->setFormat(tr("原始图像：%v / %m"));
    if (!m_rawCapturing) {
        statusBar()->showMessage(tr("原始摄像头图像采集完成：共 %1 张").arg(capturedFrames), 7000);
    }
    updateControls();
}

void MainWindow::onTrainingProgress(int percent, const QString &message)
{
    m_trainingProgress->setValue(percent);
    m_trainingProgress->setFormat(tr("%1：%p%").arg(message));
}

void MainWindow::onError(const QString &title, const QString &details)
{
    QMessageBox::critical(this, title, details);
    statusBar()->showMessage(title + QStringLiteral("：") + details, 10000);
}

void MainWindow::updateControls()
{
    const bool hasCamera = m_cameraCombo->count() > 0;
    const bool busy = m_training || m_state == ServiceState::Importing;
    m_backendCombo->setEnabled(!busy && !m_capturing);
    m_cameraCombo->setEnabled(!m_streaming && !busy);
    m_refreshButton->setEnabled(!m_streaming && !busy);
    m_startButton->setEnabled(hasCamera && !m_streaming && !busy);
    m_stopButton->setEnabled(m_streaming && !m_capturing && !busy);
    m_importButton->setEnabled(!busy && !m_capturing && !m_rawCapturing);
    m_personIdEdit->setEnabled(!busy && !m_capturing);
    m_personNameEdit->setEnabled(!busy && !m_capturing);
    m_groupCombo->setEnabled(!busy && !m_capturing);
    m_sampleCountSpin->setEnabled(!busy && !m_capturing);
    m_captureButton->setEnabled(m_streaming && !m_capturing && !m_rawCapturing && !busy);
    m_endCaptureButton->setEnabled(m_capturing);
    m_rawCaptureButton->setEnabled(m_streaming && !m_capturing && !m_rawCapturing && !busy);
    m_trainButton->setEnabled(!m_capturing && !m_rawCapturing && !busy);
    m_databaseButton->setEnabled(!m_capturing && !m_rawCapturing && !busy);
    m_recognitionCheck->setEnabled(!busy);
}

QString MainWindow::stateName(ServiceState state) const
{
    switch (state) {
    case ServiceState::Idle: return tr("空闲");
    case ServiceState::Streaming: return tr("视频运行中");
    case ServiceState::Capturing: return tr("样本采集中");
    case ServiceState::Importing: return tr("图片导入中");
    case ServiceState::Training: return tr("模型训练中");
    case ServiceState::Error: return tr("错误");
    }
    return tr("未知状态");
}
