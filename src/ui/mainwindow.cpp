#include "mainwindow.h"
#include "accessclient.h"
#include "adminlogindialog.h"

#include "facebackendservice.h"
#include "frameview.h"
#include "facedatabasedialog.h"

#include <QCheckBox>
#include <QApplication>
#include <QKeyEvent>
#include <QSlider>
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
    qApp->installEventFilter(this);
    auto *sessionTimer=new QTimer(this); sessionTimer->setInterval(100);
    connect(sessionTimer,&QTimer::timeout,this,[this] {
        if(!m_adminUnlocked) return;
        if(QDateTime::currentDateTimeUtc()>=m_adminExpires || m_activity.elapsed()>=300000) {
            lockAdministration(); return;
        }
        if(m_accessClient && m_checkClock.elapsed()>=15000) {
            m_checkClock.restart(); m_accessClient->checkSession();
        }
    }); sessionTimer->start();

    // 延迟到当前窗口构造完成后再发初始化请求，避免 backend 收到请求时控件
    // 还没有创建完。singleShot(0) 不是等待很久，而是放到下一轮事件循环。
    QTimer::singleShot(0, this, [this, autoStart] {
        const auto id=m_backendCombo->currentData().toString();
        QMetaObject::invokeMethod(m_backend,[backend=m_backend,id,autoStart] {
            backend->configureBackend(id); backend->refreshCameras();
            if(autoStart) backend->startAutomatically();
        },Qt::QueuedConnection);
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

    auto *header=new QHBoxLayout;
    auto *title=new QLabel(tr("人脸识别值守"),central);
    title->setStyleSheet("font-size:22px;font-weight:600;"); header->addWidget(title);
    m_directionLabel=new QLabel(tr("方向：等待配置"),central); header->addWidget(m_directionLabel); header->addStretch();
    m_adminButton=new QPushButton(tr("管理"),central); m_adminButton->setObjectName("adminButton");
    m_adminButton->setMinimumSize(100,44); header->addWidget(m_adminButton); rootLayout->addLayout(header);
    connect(m_adminButton,&QPushButton::clicked,this,&MainWindow::openAdministration);
    auto *sourceGroup = new QGroupBox(tr("输入与后端"), central);
    m_sourceControls=sourceGroup; sourceGroup->setObjectName("sourceControls"); sourceGroup->hide();
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
    m_managementControls=controlsScroll; controlsScroll->setObjectName("managementControls"); controlsScroll->hide();
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
    auto *voiceRow=new QHBoxLayout;
    voiceRow->addWidget(new QLabel(tr("播报音量："),modelGroup));
    m_voiceVolume=new QSlider(Qt::Horizontal,modelGroup);
    m_voiceVolume->setObjectName(QStringLiteral("voiceVolume"));
    m_voiceVolume->setRange(0,100); m_voiceVolume->setValue(90);
    auto *volumeLabel=new QLabel(QStringLiteral("90%"),modelGroup);
    m_voiceVolumeLabel=volumeLabel;
    volumeLabel->setMinimumWidth(45);
    m_voiceMute=new QCheckBox(tr("静音"),modelGroup);
    m_voiceMute->setObjectName(QStringLiteral("voiceMute"));
    voiceRow->addWidget(m_voiceVolume,1); voiceRow->addWidget(volumeLabel); voiceRow->addWidget(m_voiceMute);
    modelLayout->addLayout(voiceRow);
    connect(m_voiceVolume,&QSlider::valueChanged,this,[this,volumeLabel](int value) {
        volumeLabel->setText(QString::number(value)+"%");
        emit voiceVolumeRequested(value,m_voiceMute->isChecked());
    });
    connect(m_voiceMute,&QCheckBox::toggled,this,[this](bool muted) {
        emit voiceVolumeRequested(m_voiceVolume->value(),muted);
    });
    modelLayout->addWidget(m_trainingProgress);
    modelLayout->addWidget(m_modelLabel);
    m_stationDirection=new QComboBox(modelGroup); m_stationDirection->setObjectName("stationDirection");
    m_stationDirection->addItem(tr("本机方向：进"),"IN"); m_stationDirection->addItem(tr("本机方向：出"),"OUT");
    modelLayout->addWidget(m_stationDirection);
    connect(m_stationDirection,qOverload<int>(&QComboBox::activated),this,[this] {
        const auto direction=m_stationDirection->currentData().toString();
        withAdminAuthorization([this,direction] { if(m_accessClient) m_accessClient->updateDirection(direction); });
    });
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

    statusBar()->showMessage(tr("准备就绪，等待视频输入。"));

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
    connect(m_databaseButton,&QPushButton::clicked,this,&MainWindow::openDatabase);
    connect(m_recognitionCheck, &QCheckBox::toggled, this, &MainWindow::recognitionRequested);
    connect(m_originalView, &FrameView::faceClicked, this, [this](qint64 trackId) {
        if (m_adminUnlocked && m_capturing) {
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
    auto queue=[this,backend](std::function<void()> action) {
        withAdminAuthorization([backend,action=std::move(action)] { QMetaObject::invokeMethod(backend,action,Qt::QueuedConnection); });
    };
    connect(this,&MainWindow::configureBackendRequested,this,[queue,backend](const QString &id) { queue([backend,id] { backend->configureBackend(id); }); });
    connect(this,&MainWindow::refreshCamerasRequested,this,[queue,backend] { queue([backend] { backend->refreshCameras(); }); });
    connect(this,&MainWindow::openCameraRequested,this,[queue,backend](const QString &id) { queue([backend,id] { backend->openCamera(id); }); });
    connect(this,&MainWindow::closeInputRequested,this,[queue,backend] { queue([backend] { backend->closeInput(); }); });
    connect(this,&MainWindow::importImagesRequested,this,[queue,backend](const QStringList &paths,const PersonProfile &person,bool save) { queue([backend,paths,person,save] { backend->importImages(paths,person,save); }); });
    connect(this,&MainWindow::beginCaptureRequested,this,[queue,backend](const PersonProfile &person,int count) { queue([backend,person,count] { backend->beginCapture(person,count); }); });
    connect(this,&MainWindow::selectCaptureTrackRequested,this,[queue,backend](qint64 id) { queue([backend,id] { backend->selectCaptureTrack(id); }); });
    connect(this,&MainWindow::endCaptureRequested,this,[backend] { QMetaObject::invokeMethod(backend,&FaceBackendService::endCapture,Qt::QueuedConnection); });
    connect(this,&MainWindow::beginRawFrameCaptureRequested,this,[queue,backend] { queue([backend] { backend->beginRawFrameCapture(); }); });
    connect(this,&MainWindow::trainModelRequested,this,[queue,backend] { queue([backend] { backend->trainModel(); }); });
    connect(this,&MainWindow::recognitionRequested,this,[queue,backend](bool enabled) { queue([backend,enabled] { backend->setRecognitionEnabled(enabled); }); });
    connect(backend,&FaceBackendService::stationConfigured,this,[this](const QString &id,const QString &direction,int version,bool) {
        if(id=="board-in") setStationDirection(direction,version);
    });
    connect(this,&MainWindow::voiceVolumeRequested,backend,&FaceBackendService::setVoiceVolume,Qt::QueuedConnection);
    connect(backend,&FaceBackendService::voiceVolumeChanged,this,[this](int volume,bool muted) {
        const QSignalBlocker sliderBlock(m_voiceVolume),muteBlock(m_voiceMute);
        m_voiceVolume->setValue(volume); m_voiceMute->setChecked(muted);
        m_voiceVolumeLabel->setText(QString::number(volume)+"%");
        m_voiceVolume->setToolTip(tr("播报音量：%1%").arg(volume));
    });

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
    if(!m_adminUnlocked) return;
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

bool MainWindow::eventFilter(QObject *watched,QEvent *event)
{
    auto *widget=qobject_cast<QWidget *>(watched);
    if(m_adminUnlocked && widget && (widget==this || isAncestorOf(widget)) &&
       (event->type()==QEvent::KeyPress || event->type()==QEvent::MouseButtonPress ||
        event->type()==QEvent::TouchBegin || event->type()==QEvent::Wheel)) {
        m_activity.restart();
        if(m_accessClient && m_refreshClock.elapsed()>=15000) { m_refreshClock.restart(); m_accessClient->refreshSession(); }
    }
    if (event->type()==QEvent::KeyPress && widget && (widget==this || isAncestorOf(widget))) {
        const int key=static_cast<QKeyEvent *>(event)->key();
        if (key==Qt::Key_VolumeUp || key==Qt::Key_VolumeDown) {
            m_voiceMute->setChecked(false);
            m_voiceVolume->setValue(m_voiceVolume->value()+(key==Qt::Key_VolumeUp ? 5 : -5));
            return true;
        }
        if (key==Qt::Key_VolumeMute) {
            m_voiceMute->toggle(); return true;
        }
    }
    return QMainWindow::eventFilter(watched,event);
}

void MainWindow::startSelectedCamera()
{
    if(!m_adminUnlocked) return;
    const QString cameraId = m_cameraCombo->currentData().toString();
    if (cameraId.isEmpty()) {
        QMessageBox::warning(this, tr("未选择摄像头"), tr("请刷新并选择一个视频设备。"));
        return;
    }
    emit openCameraRequested(cameraId);
}

void MainWindow::beginCapture()
{
    if(!m_adminUnlocked) return;
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
    if(m_adminUnlocked) QMessageBox::critical(this, title, details);
    statusBar()->showMessage(title + QStringLiteral("：") + details, 10000);
}

void MainWindow::updateControls()
{
    const bool hasCamera = m_cameraCombo->count() > 0;
    const bool busy = !m_adminUnlocked || m_training || m_state == ServiceState::Importing;
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
    m_endCaptureButton->setEnabled(m_adminUnlocked && m_capturing);
    m_stationDirection->setEnabled(m_adminUnlocked);
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

void MainWindow::setAccessClient(AccessClient *client) {
    if(m_accessClient) disconnect(m_accessClient,nullptr,this,nullptr);
    m_accessClient=client;
    if(!client) return;
    connect(client,&AccessClient::sessionLost,this,&MainWindow::lockAdministration);
    connect(client,&AccessClient::sessionRefreshed,this,[this](const QDateTime &expiry) { if(m_adminUnlocked) m_adminExpires=expiry; });
    connect(client,&AccessClient::errorOccurred,this,[this](const QString &message) { statusBar()->showMessage(message,10000); });
}
void MainWindow::setAdminSession(const QString &sessionId,const QDateTime &expiresAt) {
    if(sessionId.isEmpty() || !expiresAt.isValid() || expiresAt<=QDateTime::currentDateTimeUtc()) return;
    if(m_accessClient && sessionId!=m_accessClient->sessionId()) return;
    m_adminSession=sessionId; m_adminExpires=expiresAt; m_adminUnlocked=true;
    m_activity.start(); m_refreshClock.start(); m_checkClock.start();
    m_sourceControls->show(); m_managementControls->show(); m_adminButton->setText(tr("返回值守")); updateControls();
}
void MainWindow::lockAdministration() {
    const bool wasUnlocked=m_adminUnlocked;
    m_adminUnlocked=false; m_adminSession.clear(); m_adminExpires={};
    m_sourceControls->hide(); m_managementControls->hide(); m_adminButton->setText(tr("管理"));
    if(m_database) { m_database->close(); m_database->deleteLater(); m_database=nullptr; }
    if(m_login) { m_login->reject(); m_login->deleteLater(); m_login=nullptr; }
    if(wasUnlocked) emit endCaptureRequested();
    updateControls();
    if(m_accessClient && !m_accessClient->sessionId().isEmpty()) m_accessClient->logout();
}
void MainWindow::withAdminAuthorization(std::function<void()> action) {
    if(!m_adminUnlocked || QDateTime::currentDateTimeUtc()>=m_adminExpires || m_activity.elapsed()>=300000) { lockAdministration(); return; }
    const auto session=m_adminSession;
    if(m_accessClient) m_accessClient->authorizeAction([this,session,action=std::move(action)] {
        if(m_adminUnlocked && session==m_adminSession && QDateTime::currentDateTimeUtc()<m_adminExpires) action();
    });
    else action(); // A controlled session can be supplied by standalone UI hosts.
}
void MainWindow::openAdministration() {
    if(m_adminUnlocked) { lockAdministration(); return; }
    if(m_login) { m_login->raise(); return; }
    m_login=new AdminLoginDialog(m_accessClient,this); auto *dialog=m_login.data();
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    connect(dialog,&QDialog::accepted,this,[this] {
        if(m_accessClient) setAdminSession(m_accessClient->sessionId(),m_accessClient->expiresAt());
    }); dialog->open();
}
void MainWindow::openDatabase() {
    if(!m_adminUnlocked) return;
    withAdminAuthorization([this] {
        if(!m_database) {
            m_database=new FaceDatabaseDialog(m_backend,this);
            m_database->setAuthorization([this](std::function<void()> action) { withAdminAuthorization(std::move(action)); });
            connect(m_database,&FaceDatabaseDialog::capturePersonSelected,this,[this](const PersonProfile &person) {
                if(!m_adminUnlocked) return;
                m_personIdEdit->setText(person.personId); m_personNameEdit->setText(person.displayName);
                m_groupCombo->setCurrentIndex(m_groupCombo->findData(person.group));
            });
        } m_database->open();
    });
}
void MainWindow::setStationDirection(const QString &direction,int version) {
    const QSignalBlocker blocker(m_stationDirection);
    m_stationDirection->setCurrentIndex(m_stationDirection->findData(direction));
    m_directionLabel->setText(tr("方向：%1（版本%2）").arg(direction=="IN" ? tr("进") : tr("出")).arg(version));
}
