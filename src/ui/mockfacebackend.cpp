#include "mockfacebackend.h"

#include <QDateTime>
#include <QImageReader>
#include <QLinearGradient>
#include <QPainter>
#include <QTimer>
#include <QtMath>

MockFaceBackend::MockFaceBackend(QObject *parent)
    : FaceBackendService(parent)
    , m_frameTimer(new QTimer(this))
    , m_trainingTimer(new QTimer(this))
{
    m_frameTimer->setInterval(33);
    m_frameTimer->setTimerType(Qt::PreciseTimer);
    connect(m_frameTimer, &QTimer::timeout, this, &MockFaceBackend::generateFrame);

    m_trainingTimer->setInterval(120);
    connect(m_trainingTimer, &QTimer::timeout, this, &MockFaceBackend::advanceTraining);
}

void MockFaceBackend::requestShutdown() noexcept
{
    m_shutdownRequested.store(true, std::memory_order_relaxed);
}

void MockFaceBackend::configureBackend(const QString &backendId)
{
    if (backendId != QStringLiteral("opencv_lbph")
        && backendId != QStringLiteral("onnx_embedding")
        && backendId != QStringLiteral("mobilefacenet_rknn")) {
        emit errorOccurred(tr("后端配置失败"), tr("不支持的后端：%1").arg(backendId));
        return;
    }
    m_backendId = backendId;
    emit modelChanged(QStringLiteral("demo-untrained"),
                      backendId == QStringLiteral("opencv_lbph")
                          ? tr("已选择 OpenCV / LBPH 后端")
                          : backendId == QStringLiteral("mobilefacenet_rknn")
                              ? tr("已选择 RKNN / MobileFaceNet 后端")
                              : tr("已选择 ONNX 特征后端"));
}

void MockFaceBackend::refreshCameras()
{
    CameraList cameras;
    cameras.push_back({QStringLiteral("camera:0"), tr("摄像头 0（演示）")});
    cameras.push_back({QStringLiteral("camera:1"), tr("摄像头 1（演示）")});
    emit camerasChanged(cameras);
    setState(m_running ? ServiceState::Streaming : ServiceState::Idle,
             tr("发现 %1 个视频设备").arg(cameras.size()));
}

void MockFaceBackend::startAutomatically()
{
    openCamera(QStringLiteral("camera:0"));
    setRecognitionEnabled(true);
}

void MockFaceBackend::openCamera(const QString &cameraId)
{
    if (cameraId.isEmpty()) {
        emit errorOccurred(tr("无法启动摄像头"), tr("请先选择视频设备。"));
        return;
    }
    m_cameraId = cameraId;
    m_running = true;
    m_sequence = 0;
    m_frameTimer->start();
    setState(ServiceState::Streaming, tr("正在读取 %1（模拟画面）").arg(cameraId));
}

void MockFaceBackend::closeInput()
{
    m_frameTimer->stop();
    m_running = false;
    m_captureActive = false;
    m_rawCaptureActive = false;
    m_rawCaptureAccepted = 0;
    m_rawCaptureTarget = 0;
    emit captureProgress(0, 0);
    emit rawCaptureProgress(0, 0);
    setState(ServiceState::Idle, tr("视频输入已停止"));
}

void MockFaceBackend::importImages(const QStringList &paths,
                                   const PersonProfile &person,
                                   bool saveToDataset)
{
    Q_UNUSED(person)
    Q_UNUSED(saveToDataset)
    if (paths.isEmpty()) {
        return;
    }

    setState(ServiceState::Importing, tr("正在导入 %1 张图片").arg(paths.size()));
    QImageReader reader(paths.constFirst());
    reader.setAutoTransform(true);
    const QImage image = reader.read();
    if (image.isNull()) {
        setState(m_running ? ServiceState::Streaming : ServiceState::Idle, tr("图片导入失败"));
        emit errorOccurred(tr("无法读取图片"),
                           tr("文件：%1\n原因：%2").arg(paths.constFirst(), reader.errorString()));
        return;
    }

    FrameResult frame;
    frame.originalImage = image.convertToFormat(QImage::Format_RGB32);
    frame.processedImage = frame.originalImage;
    frame.sequence = ++m_sequence;
    emit frameReady(frame);
    setState(m_running ? ServiceState::Streaming : ServiceState::Idle,
             tr("已导入 %1 张图片；原型当前显示第一张").arg(paths.size()));
}

void MockFaceBackend::selectCaptureTrack(qint64 trackId)
{
    if (m_captureActive) {
        setState(ServiceState::Capturing, tr("已选择模拟目标 #%1").arg(trackId));
    }
}

void MockFaceBackend::beginCapture(const PersonProfile &person, int targetSamples)
{
    if (!m_running) {
        emit errorOccurred(tr("无法开始采集"), tr("请先启动摄像头。"));
        return;
    }
    if (person.personId.isEmpty() || person.displayName.isEmpty() || targetSamples <= 0) {
        emit errorOccurred(tr("人员信息不完整"), tr("请填写人员 ID、姓名和有效的采集数量。"));
        return;
    }

    m_capturePerson = person;
    m_captureAccepted = 0;
    m_captureTarget = targetSamples;
    m_captureActive = true;
    emit captureProgress(0, m_captureTarget);
    setState(ServiceState::Capturing,
             tr("正在为 %1 采集合格人脸").arg(person.displayName));
}

void MockFaceBackend::endCapture()
{
    if (!m_captureActive) {
        return;
    }
    m_captureActive = false;
    setState(m_running ? ServiceState::Streaming : ServiceState::Idle,
             tr("采集已停止：%1/%2").arg(m_captureAccepted).arg(m_captureTarget));
}

void MockFaceBackend::beginRawFrameCapture()
{
    if (!m_running) {
        emit errorOccurred(tr("无法采集原始图像"), tr("请先启动摄像头。"));
        return;
    }
    if (m_captureActive || m_training) {
        emit errorOccurred(tr("无法采集原始图像"),
                           tr("请先结束当前人脸采集或训练任务。"));
        return;
    }
    if (m_rawCaptureActive) {
        return;
    }
    m_rawCaptureAccepted = 0;
    m_rawCaptureTarget = 20;
    m_rawCaptureActive = true;
    emit rawCaptureProgress(0, m_rawCaptureTarget);
    setState(ServiceState::Streaming, tr("正在采集原始摄像头图像：0/20"));
}

void MockFaceBackend::trainModel()
{
    if (m_training) {
        emit errorOccurred(tr("训练正在进行"), tr("请等待当前训练任务完成。"));
        return;
    }
    if (m_captureActive) {
        emit errorOccurred(tr("暂不能训练"), tr("请先结束人脸采集。"));
        return;
    }

    m_training = true;
    m_trainingPercent = 0;
    emit trainingProgress(0, tr("检查数据集"));
    setState(ServiceState::Training, tr("正在训练模型（模拟）"));
    m_trainingTimer->start();
}

void MockFaceBackend::setRecognitionEnabled(bool enabled)
{
    m_recognitionEnabled = enabled;
    emit recognitionEnabledChanged(enabled);
    setState(m_captureActive ? ServiceState::Capturing
                             : (m_running ? ServiceState::Streaming : ServiceState::Idle),
             enabled ? tr("实时识别已开启") : tr("仅进行人脸检测"));
}

void MockFaceBackend::shutdown()
{
    m_shutdownRequested.store(true, std::memory_order_relaxed);
    m_frameTimer->stop();
    m_trainingTimer->stop();
    m_running = false;
    m_captureActive = false;
    m_rawCaptureActive = false;
    m_rawCaptureAccepted = 0;
    m_rawCaptureTarget = 0;
    m_training = false;
    emit rawCaptureProgress(0, 0);
}

void MockFaceBackend::generateFrame()
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) {
        return;
    }
    FrameResult frame = makeDemoFrame();
    emit frameReady(frame);

    if (m_captureActive && (m_sequence % 5 == 0)) {
        ++m_captureAccepted;
        emit captureProgress(m_captureAccepted, m_captureTarget);
        if (m_captureAccepted >= m_captureTarget) {
            m_captureActive = false;
            setState(ServiceState::Streaming,
                     tr("采集完成：%1，共 %2 张")
                         .arg(m_capturePerson.displayName)
                         .arg(m_captureAccepted));
        }
    }

    if (m_rawCaptureActive && (m_sequence % 5 == 0)) {
        ++m_rawCaptureAccepted;
        emit rawCaptureProgress(m_rawCaptureAccepted, m_rawCaptureTarget);
        if (m_rawCaptureAccepted >= m_rawCaptureTarget) {
            m_rawCaptureActive = false;
            setState(ServiceState::Streaming,
                     tr("原始摄像头图像采集完成：共 %1 张").arg(m_rawCaptureAccepted));
        }
    }
}

void MockFaceBackend::advanceTraining()
{
    if (m_shutdownRequested.load(std::memory_order_relaxed)) {
        m_trainingTimer->stop();
        return;
    }
    m_trainingPercent = qMin(100, m_trainingPercent + 10);
    emit trainingProgress(m_trainingPercent,
                          m_trainingPercent < 40 ? tr("预处理训练样本")
                          : m_trainingPercent < 80 ? tr("训练识别器")
                                                   : tr("标定拒识阈值"));
    if (m_trainingPercent < 100) {
        return;
    }

    m_trainingTimer->stop();
    m_training = false;
    const QString version = QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd-HHmmss"));
    emit modelChanged(version,
                      m_backendId == QStringLiteral("opencv_lbph")
                          ? tr("LBPH 演示模型已就绪")
                          : m_backendId == QStringLiteral("mobilefacenet_rknn")
                              ? tr("MobileFaceNet 特征库演示版本已就绪")
                              : tr("ONNX 特征库演示版本已就绪"));
    setState(m_running ? ServiceState::Streaming : ServiceState::Idle,
             tr("训练完成（模拟）"));
}

FrameResult MockFaceBackend::makeDemoFrame()
{
    constexpr int width = 960;
    constexpr int height = 540;
    FrameResult frame;
    frame.sequence = ++m_sequence;
    frame.fps = 30.0;
    frame.originalImage = QImage(width, height, QImage::Format_RGB32);

    QPainter painter(&frame.originalImage);
    QLinearGradient background(0, 0, width, height);
    background.setColorAt(0.0, QColor("#1e293b"));
    background.setColorAt(1.0, QColor("#0f172a"));
    painter.fillRect(frame.originalImage.rect(), background);
    painter.setPen(QColor("#cbd5e1"));
    painter.setFont(QFont(QString(), 18, QFont::DemiBold));
    painter.drawText(QRect(24, 18, width - 48, 32),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     tr("UI 原型模拟视频 · %1").arg(m_cameraId));

    const qreal oscillation = qSin(static_cast<qreal>(m_sequence) / 35.0) * 22.0;
    const QRectF firstFace(155.0 + oscillation, 145.0, 190.0, 230.0);
    const QRectF secondFace(590.0 - oscillation, 170.0, 175.0, 215.0);

    auto drawDemoPerson = [&painter](const QRectF &face, const QColor &shirt) {
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor("#e8b38f"));
        painter.drawEllipse(face.adjusted(22, 0, -22, -55));
        painter.setBrush(QColor("#172033"));
        painter.drawPie(face.adjusted(20, -5, -20, -105), 0, 180 * 16);
        painter.setBrush(shirt);
        painter.drawRoundedRect(QRectF(face.left() - 20, face.bottom() - 70,
                                       face.width() + 40, 140), 35, 35);
    };
    drawDemoPerson(firstFace, QColor("#0369a1"));
    drawDemoPerson(secondFace, QColor("#7c3aed"));
    painter.end();

    FaceObservation first;
    first.normalizedBox = QRectF(firstFace.left() / width, firstFace.top() / height,
                                 firstFace.width() / width, firstFace.height() / height);
    first.trackId = 1;
    first.color = QColor("#22c55e");

    FaceObservation second;
    second.normalizedBox = QRectF(secondFace.left() / width, secondFace.top() / height,
                                  secondFace.width() / width, secondFace.height() / height);
    second.trackId = 2;
    second.color = QColor("#f59e0b");

    if (m_recognitionEnabled) {
        first.identityState = IdentityState::Known;
        first.personId = QStringLiteral("IN_001");
        first.displayName = tr("演示人员");
        first.group = tr("组内");
        first.scoreText = m_backendId == QStringLiteral("opencv_lbph")
                              ? tr("距离 38.4")
                              : tr("相似度 0.86");
        second.identityState = IdentityState::Unknown;
        second.scoreText = m_backendId == QStringLiteral("opencv_lbph")
                               ? tr("距离 91.7")
                               : tr("相似度 0.31");
    }

    frame.faces = {first, second};
    frame.processedImage = frame.originalImage;
    return frame;
}

void MockFaceBackend::setState(ServiceState state, const QString &message)
{
    emit stateChanged(state, message);
}
