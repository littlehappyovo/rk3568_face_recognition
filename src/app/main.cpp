#include "backendtypes.h"
#include "mainwindow.h"
#include "mockfacebackend.h"
#include "opencvfacebackend.h"

#include <facecore/facecore.hpp>

#include <QApplication>
#include <QCoreApplication>
#include <QDateTime>
#include <QDebug>
#include <QDir>
#include <QMetaObject>
#include <QScreen>
#include <QStandardPaths>
#include <QString>
#include <QStringList>
#include <QTemporaryDir>
#include <QAbstractButton>
#include <QEvent>
#include <QFont>
#include <QMouseEvent>
#include <QTouchEvent>
#include <QTimer>
#include <QWidget>
#include <QThread>

#include <cstring>
#include <filesystem>

namespace {

// main.cpp 只做三件事：建立 QApplication/事件循环，创建界面和业务对象，
// 最后在程序退出时按正确顺序停止工作线程。真正的人脸算法不应写在这里。

// --log-input 时装的调试事件过滤器：把每次点击/触摸及其命中控件打进日志。
// 板端是 MIPI 触摸屏 + eglfs 旋转，而 QT_QPA_EGLFS_ROTATION 只转画面不转触摸坐标；
// 有这个日志就能看出坐标被映射到哪、命中了哪个控件，从而算出正确的校准矩阵。
class InputLogger : public QObject
{
public:
    explicit InputLogger(QObject *parent = nullptr) : QObject(parent) {}

protected:
    bool eventFilter(QObject *watched, QEvent *event) override
    {
        const char *watchedClass =
            watched != nullptr ? watched->metaObject()->className() : "?";
        // 带上墙钟毫秒，方便和板端 python 抓的 evdev 原始触摸流对齐。
        const QString stamp =
            QStringLiteral("[input %1]").arg(QDateTime::currentMSecsSinceEpoch());
        switch (event->type()) {
        case QEvent::MouseButtonPress:
        case QEvent::MouseButtonRelease: {
            auto *mouse = static_cast<QMouseEvent *>(event);
            const QWidget *hit = QApplication::widgetAt(mouse->globalPos());
            qInfo().noquote()
                << stamp
                << "mouse"
                << (event->type() == QEvent::MouseButtonPress ? "press" : "release")
                << "pos=" << mouse->pos() << "global=" << mouse->globalPos() << "target="
                << watchedClass << "widgetAt="
                << (hit != nullptr ? hit->metaObject()->className() : "none");
            break;
        }
        case QEvent::TouchBegin:
        case QEvent::TouchUpdate:
        case QEvent::TouchEnd: {
            auto *touch = static_cast<QTouchEvent *>(event);
            for (const QTouchEvent::TouchPoint &point : touch->touchPoints()) {
                // Qt 5.15 的 TouchPoint 只有 screenPos()，没有 globalPos()。
                qInfo().noquote() << stamp << "touch type=" << event->type() << "id="
                                  << point.id()
                                  << "pos=" << point.pos() << "screen=" << point.screenPos()
                                  << "state=" << point.state();
            }
            break;
        }
        default:
            break;
        }
        return false;  // 只看不拦
    }
};


bool rawArgumentsContain(int argc, char *argv[], const char *option)
{
    for (int index = 1; index < argc; ++index) {
        if (argv[index] != nullptr && std::strcmp(argv[index], option) == 0) {
            return true;
        }
    }
    return false;
}

// 取 --name value 形式的命令行取值，缺省返回空串。
QString optionValue(const QStringList &arguments, const QString &name)
{
    const int index = arguments.indexOf(name);
    if (index < 0 || index + 1 >= arguments.size()) {
        return {};
    }
    return arguments.at(index + 1);
}

int runSelfTest(const QStringList &arguments)
{
    QTemporaryDir temporaryData;
    if (!temporaryData.isValid()) {
        qCritical() << "Cannot create temporary self-test data directory";
        return 1;
    }
    qputenv("FACEAPP_DATA_DIR", temporaryData.path().toLocal8Bit());

    const QString configPath = optionValue(arguments, QStringLiteral("--config"));
    if (!configPath.isEmpty()) {
        qputenv("FACEAPP_CONFIG", configPath.toLocal8Bit());
        qInfo().noquote() << "self-test config:" << configPath;
    }

    qInfo().noquote() << "build capabilities:"
                      << QString::fromStdString(facecore::describeBackendCapabilities());
    const facecore::RknnRuntimeInfo runtimeInfo = facecore::probeRknnRuntime();
    const QString runtimeText = QString::fromStdString(facecore::describeRknnRuntime(runtimeInfo));
    for (const QString &line : runtimeText.split(QLatin1Char('\n'), Qt::SkipEmptyParts)) {
        qInfo().noquote() << "  " << line;
    }

    OpenCvFaceBackend backend(QCoreApplication::applicationDirPath());
    int exitCode = 0;
    int reportedErrors = 0;
    QObject::connect(&backend, &FaceBackendService::errorOccurred,
                     [&exitCode, &reportedErrors](const QString &title, const QString &details) {
                         qCritical().noquote() << title << details;
                         ++reportedErrors;
                         exitCode = 1;
                     });
    QObject::connect(&backend, &FaceBackendService::camerasChanged,
                     [](const CameraList &cameras) {
                         for (const CameraDescriptor &camera : cameras) {
                             qInfo().noquote() << "camera candidate:" << camera.id
                                               << camera.displayName;
                         }
                     });
    QObject::connect(&backend, &FaceBackendService::modelChanged,
                     [](const QString &state, const QString &summary) {
                         qInfo().noquote() << "backend state:" << state << "|" << summary;
                     });

    // 1) LBPH 通路：不需要任何外部模型，必须在任何环境上通过。
    backend.configureBackend(QStringLiteral("opencv_lbph"));
    // 2) embedding 通路：有 .rknn 就走 NPU，否则回退 CPU；两者都没有时只提示“模型待投放”。
    backend.configureBackend(QStringLiteral("onnx_embedding"));
    // 3) 默认的 RK3568 识别通路：MobileFaceNet 由 NPU 执行。
    backend.configureBackend(QStringLiteral("mobilefacenet_rknn"));
    // 4) 摄像头枚举（只读，不打开设备）。
    backend.refreshCameras();

    if (exitCode == 0) {
        qInfo() << "FaceRecognitionSystem self-test passed";
    } else {
        qCritical() << "FaceRecognitionSystem self-test reported" << reportedErrors << "error(s)";
    }
    return exitCode;
}

// --log-input 时把界面实测几何打进日志。旋转后如果"看到的按钮"和 Qt 认为的
// 按钮位置不一致（触摸坐标映射错），这份几何就是分析用的基准数据。
void dumpUiGeometry(QWidget *root)
{
    if (root == nullptr) {
        return;
    }
    const QRect geometry = root->geometry();
    const QRect frame = root->frameGeometry();
    qInfo().noquote() << QStringLiteral("[ui] window class=%1 geometry=%2x%3+%4+%5 frame=%6x%7")
                             .arg(QString::fromLatin1(root->metaObject()->className()))
                             .arg(geometry.width()).arg(geometry.height())
                             .arg(geometry.x()).arg(geometry.y())
                             .arg(frame.width()).arg(frame.height());

    const QStringList wanted = {
        QStringLiteral("QAbstractButton"), QStringLiteral("QGroupBox"),
        QStringLiteral("QComboBox"),       QStringLiteral("QLineEdit"),
        QStringLiteral("QSpinBox"),        QStringLiteral("QProgressBar"),
        QStringLiteral("QScrollArea"),     QStringLiteral("QSplitter"),
        QStringLiteral("QAbstractScrollArea")};

    const QList<QWidget *> widgets = root->findChildren<QWidget *>();
    int buttons = 0;
    int offScreen = 0;
    int covered = 0;
    int disabled = 0;
    const QRect screen = root->screen() != nullptr ? root->screen()->geometry()
                                                  : QRect(0, 0, root->width(), root->height());
    for (QWidget *widget : widgets) {
        if (widget == nullptr || widget->isWindow()) {
            continue;
        }
        bool interesting = false;
        for (const QString &base : wanted) {
            if (widget->inherits(base.toLatin1().constData())) {
                interesting = true;
                break;
            }
        }
        if (!interesting) {
            continue;
        }
        const QPoint topLeft = widget->mapToGlobal(QPoint(0, 0));
        const QPoint center = widget->mapToGlobal(widget->rect().center());
        QWidget *hit = QApplication::widgetAt(center);
        QString label = widget->objectName();
        if (auto *button = qobject_cast<QAbstractButton *>(widget)) {
            label = button->text();
        }
        qInfo().noquote() << QStringLiteral("[ui] %1 \"%2\" global=%3,%4 size=%5x%6 center=%7,%8 enabled=%9 visible=%10 hint=%11x%12 minHint=%13x%14 fontPx=%15 fm=%16 widgetAt=%17")
                                 .arg(QString::fromLatin1(widget->metaObject()->className()))
                                 .arg(label)
                                 .arg(topLeft.x()).arg(topLeft.y())
                                 .arg(widget->width()).arg(widget->height())
                                 .arg(center.x()).arg(center.y())
                                 .arg(widget->isEnabled() ? 1 : 0)
                                 .arg(widget->isVisible() ? 1 : 0)
                                 .arg(widget->sizeHint().width())
                                 .arg(widget->sizeHint().height())
                                 .arg(widget->minimumSizeHint().width())
                                 .arg(widget->minimumSizeHint().height())
                                 .arg(widget->font().pixelSize())
                                 .arg(widget->fontMetrics().height())
                                 .arg(hit != nullptr
                                          ? QString::fromLatin1(hit->metaObject()->className())
                                          : QStringLiteral("null"));
        if (qobject_cast<QAbstractButton *>(widget) != nullptr) {
            ++buttons;
            if (!screen.contains(center)) {
                ++offScreen;
            } else if (hit != widget) {
                ++covered;
            }
            if (!widget->isEnabled()) {
                ++disabled;
            }
        }
    }
    qInfo().noquote() << QStringLiteral("[ui] check screen=%1x%2 buttons=%3 offscreen=%4 covered=%5 disabled=%6")
                             .arg(screen.width()).arg(screen.height())
                             .arg(buttons).arg(offScreen).arg(covered).arg(disabled);
}

} // namespace

int main(int argc, char *argv[])
{
    // QApplication 构造时就会加载 Qt 平台插件，所以 --self-test 必须在
    // 构造 QApplication 之前设置 offscreen，才能在没有 DISPLAY 的 SSH 会话中运行。
    // QApplication loads the platform plugin in its constructor. Select the
    // headless plugin before construction so a bare SSH `--self-test` works.
    if (rawArgumentsContain(argc, argv, "--self-test") &&
        qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    QApplication application(argc, argv);
    QCoreApplication::setApplicationName(QStringLiteral("FaceRecognitionSystem"));
    QCoreApplication::setApplicationVersion(QStringLiteral("0.2.0"));
    QCoreApplication::setOrganizationName(QStringLiteral("PortableFaceSystem"));
    registerFaceBackendMetaTypes();

    // 调试用：把每次点击/触摸及命中控件打进日志，用于排查旋转后的触摸坐标映射。
    if (rawArgumentsContain(argc, argv, "--log-input")) {
        qInfo().noquote() << "[input] 输入日志已启用（--log-input）";
        application.installEventFilter(new InputLogger(&application));
    }

    // 板端是 MIPI 竖屏 1080x1920，而界面按 1440x850 的宽屏布局设计；
    // 竖着放会横向被裁（控件最小宽度之和就超 1080）。启动脚本用
    // QT_QPA_EGLFS_ROTATION=90 把画面转成 1920x1080 横向，这里把实际拿到
    // 多少尺寸打出来，方便远程确认旋转到底生效没有。
    {
        const QScreen *screen = QGuiApplication::primaryScreen();
        if (screen != nullptr) {
            const QRect geometry = screen->geometry();
            qInfo().noquote() << QStringLiteral("[display] screen=%1x%2 available=%3x%4 dpr=%5")
                                     .arg(geometry.width())
                                     .arg(geometry.height())
                                     .arg(screen->availableGeometry().width())
                                     .arg(screen->availableGeometry().height())
                                     .arg(screen->devicePixelRatio());
        } else {
            qWarning() << "[display] 没有拿到 primaryScreen";
        }
        const QString platform = QGuiApplication::platformName();
        qInfo().noquote() << "[display] platform =" << platform
                          << " rotation env =" << qgetenv("QT_QPA_EGLFS_ROTATION");
    }

    // 板端 eglfs 会按面板物理尺寸反推 DPI：MIPI 5.5 寸屏算出来 400+ DPI，
    // Qt 默认字号跟着放大四五倍，界面于是撑爆 1920x1080——控件互相重叠、
    // 右栏要滚动才看得到，手指按"看到的按钮"经常按在别的控件或者空白区，
    // 表现就是"按钮点不动"。这里用像素字号把界面拉回正常尺寸：像素字号
    // 不随 DPI 变化，1920x1080 上 20px 既清楚又适合触摸。
    {
        QFont font = QApplication::font();
        const int pixelSize = qEnvironmentVariableIntValue("FACEAPP_UI_FONT_PX");
        font.setPixelSize(pixelSize > 0 ? pixelSize : 20);
        const QByteArray family = qgetenv("FACEAPP_UI_FONT_FAMILY");
        if (!family.isEmpty()) {
            font.setFamily(QString::fromLocal8Bit(family));
        }
        QApplication::setFont(font);
        // 光改字号不够：eglfs 报的 471.9 DPI 还会让样式里按 DPI 换算的那几个
        // 尺寸（输入框/微调框的内边距等）一起放大，编辑框长到 123px 高，
        // 表单被撑成两三倍。这里用像素值把关键控件的高度钉死，几何就可预期了。
        application.setStyleSheet(QStringLiteral(
            "QLineEdit, QSpinBox, QComboBox { min-height: 34px; max-height: 42px; }"
            "QPushButton, QToolButton { min-height: 42px; max-height: 54px; padding: 2px 10px; }"
            "QCheckBox { min-height: 32px; max-height: 40px; }"
            "QProgressBar { min-height: 26px; max-height: 32px; }"));
        const QScreen *screen = QGuiApplication::primaryScreen();
        qInfo().noquote()
            << QStringLiteral("[ui-font] family=%1 pixelSize=%2 logicalDpi=%3 physicalDpi=%4 physicalSize=%5x%6mm")
                   .arg(font.family())
                   .arg(font.pixelSize())
                   .arg(screen != nullptr ? screen->logicalDotsPerInch() : -1.0, 0, 'f', 1)
                   .arg(screen != nullptr ? screen->physicalDotsPerInch() : -1.0, 0, 'f', 1)
                   .arg(screen != nullptr ? screen->physicalSize().width() : -1.0, 0, 'f', 1)
                   .arg(screen != nullptr ? screen->physicalSize().height() : -1.0, 0, 'f', 1);
    }

    const QStringList arguments = application.arguments();
    // --config 对所有模式都生效（OpenCvFaceBackend 通过 FACEAPP_CONFIG 读取）。
    const QString configOption = optionValue(arguments, QStringLiteral("--config"));
    if (!configOption.isEmpty()) {
        qputenv("FACEAPP_CONFIG", configOption.toLocal8Bit());
    }
    if (arguments.contains(QStringLiteral("--self-test"))) {
        return runSelfTest(arguments);
    }

    const bool useMock = arguments.contains(QStringLiteral("--mock"));
    FaceBackendService *backend = useMock
        ? static_cast<FaceBackendService *>(new MockFaceBackend)
        : static_cast<FaceBackendService *>(
              new OpenCvFaceBackend(QCoreApplication::applicationDirPath()));

    // 后端包含摄像头读取、NPU/CPU 推理和磁盘操作，移动到独立线程可以避免
    // 主线程被这些耗时任务阻塞。QWidget 仍然只留在主线程。
    QThread backendThread;
    backendThread.setObjectName(QStringLiteral("FaceBackendThread"));
    backend->moveToThread(&backendThread);
    QObject::connect(&backendThread, &QThread::finished,
                     backend, &QObject::deleteLater);
    backendThread.start();

    // Default launch opens the configured camera and the selected valid gallery.
    // --auto-start remains compatible with existing launchers; --no-auto-start
    // explicitly opts out for maintenance or an interactive camera selection.
    MainWindow window(backend, !arguments.contains(QStringLiteral("--no-auto-start")));
    if (arguments.contains(QStringLiteral("--fullscreen"))) {
        // MIPI 屏是 1080x1920 竖屏，用全屏铺满，避免窗口比屏幕还大。
        window.showFullScreen();
    } else {
        window.show();
    }

    if (rawArgumentsContain(argc, argv, "--log-input")) {
        // 等布局稳定（eglfs 全屏 + 分割条初始化）后再量一次控件几何。
        QTimer::singleShot(1500, &window, [&window] { dumpUiGeometry(&window); });
    }

    // exec() 进入 Qt 事件循环：按钮点击、QTimer 超时和 queued signal 都在这里
    // 被分发。它返回后才进入下面的有序关闭流程。
    const int result = application.exec();

    backend->requestShutdown();
    QMetaObject::invokeMethod(backend, "shutdown", Qt::BlockingQueuedConnection);
    backendThread.quit();
    backendThread.wait();
    return result;
}
