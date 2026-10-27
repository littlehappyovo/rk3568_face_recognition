#include "accessruntime.h"
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QLockFile>
#include <QProcessEnvironment>
#include <QRandomGenerator>
#include <algorithm>

AccessRuntime::AccessRuntime(QString applicationDirectory, QObject *parent)
    : QObject(parent), m_applicationDirectory(std::move(applicationDirectory))
{
    for (int index=0; index<8; ++index)
        m_secret += QByteArray::number(QRandomGenerator::system()->generate(), 16).rightJustified(8, '0');
    m_restart.setSingleShot(true);
    m_configTimer.setInterval(5000);
    connect(&m_configTimer, &QTimer::timeout, this, &AccessRuntime::synchronizeStations);
    connect(&m_restart, &QTimer::timeout, this, &AccessRuntime::launch);
    connect(&m_process, &QProcess::started, this, [this] {
        const QJsonObject privateConfig{{"local_secret", QString::fromLatin1(m_secret)},
            {"qt_socket", QDir(m_directory).filePath(QStringLiteral("control.sock"))}};
        m_process.write(QJsonDocument(privateConfig).toJson(QJsonDocument::Compact) + '\n');
        m_process.closeWriteChannel();
    });
    connect(&m_process, &QProcess::readyReadStandardError, this, [this] {
        // Never forward a child traceback or arbitrary request content into the UI.
        m_process.readAllStandardError();
    });
    connect(&m_process, &QProcess::readyReadStandardOutput, this, [this] {
        m_output += m_process.readAllStandardOutput();
        if (m_output.size() > 65536) { m_output.clear(); return; }
        for (int newline = m_output.indexOf('\n'); newline >= 0; newline = m_output.indexOf('\n')) {
            const auto value = QJsonDocument::fromJson(m_output.left(newline)).object();
            m_output.remove(0, newline+1);
            const int port = value.value(QStringLiteral("port")).toInt();
            if (value.value(QStringLiteral("status")).toString() == QStringLiteral("ready")
                && port > 0 && port <= 65535 && (m_port == 0 || port == m_port)) {
                m_backoff = 1000;
                m_baseUrl = QUrl(QStringLiteral("http://127.0.0.1:%1").arg(port));
                emit ready(m_baseUrl);
                m_configTimer.start(); synchronizeStations();
            }
        }
    });
    connect(&m_process, QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished), this,
        [this](int, QProcess::ExitStatus) { scheduleRestart(); });
    connect(&m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart) scheduleRestart();
    });
}

AccessRuntime::~AccessRuntime() { stop(); }
bool AccessRuntime::isRunning() const { return m_process.state() != QProcess::NotRunning; }

void AccessRuntime::start(const QString &runtimeDirectory, const QString &pythonExecutable)
{
    if (!m_stopping) return;
    m_directory = QDir(runtimeDirectory).absolutePath();
    if (!QDir().mkpath(m_directory) || !QFile::setPermissions(m_directory,
        QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner)) {
        emit fault(tr("管理后台无法建立私有运行目录")); return;
    }
    m_lock = std::make_unique<QLockFile>(QDir(m_directory).filePath(QStringLiteral("runtime.lock")));
    if (!m_lock->tryLock(0)) { emit fault(tr("管理后台已有其他实例运行")); return; }
    m_python = pythonExecutable;
    if (m_python.isEmpty()) m_python = QString::fromLocal8Bit(qgetenv("FACEAPP_ACCESS_PYTHON"));
    if (m_python.isEmpty()) {
        for (const QString &relative : {QStringLiteral("../server/.venv/bin/python"), QStringLiteral("server/.venv/bin/python")}) {
            const QString candidate = QDir(m_applicationDirectory).absoluteFilePath(relative);
            if (QFileInfo::exists(candidate)) { m_python = candidate; break; }
        }
    }
    bool validPort = false;
    const int configuredPort = qEnvironmentVariableIntValue("FACEAPP_ACCESS_PORT", &validPort);
    m_port = validPort ? configuredPort : 8080;
    if (m_python.isEmpty() || m_port < 0 || m_port > 65535) {
        m_lock.reset(); emit fault(tr("管理后台缺少Python环境或端口配置无效，请运行依赖安装脚本")); return;
    }
    m_stopping = false; m_backoff = 1000; launch();
}

void AccessRuntime::launch()
{
    if (m_stopping || isRunning()) return;
    m_output.clear();
    auto environment = QProcessEnvironment::systemEnvironment();
    environment.remove(QStringLiteral("PYTHONPATH")); environment.remove(QStringLiteral("PYTHONHOME"));
    m_process.setProcessEnvironment(environment);
    m_process.start(m_python, {QStringLiteral("-I"),
        QDir(m_applicationDirectory).filePath(QStringLiteral("server/start_server.py")),
        QStringLiteral("--runtime-dir"), m_directory,
        QStringLiteral("--port"), QString::number(m_port)});
}

void AccessRuntime::scheduleRestart()
{
    if (m_stopping || m_restart.isActive()) return;
    m_configTimer.stop();
    emit fault(tr("管理后台未就绪；摄像头继续运行，记录保留等待同步"));
    m_restart.start(m_backoff); m_backoff = std::min(30000, m_backoff*2);
}

void AccessRuntime::stop()
{
    m_stopping = true; m_restart.stop(); m_configTimer.stop();
    if (m_configReply) m_configReply->abort();
    if (isRunning()) {
        m_process.terminate();
        if (!m_process.waitForFinished(3000)) { m_process.kill(); m_process.waitForFinished(1000); }
    }
    m_lock.reset();
}

void AccessRuntime::synchronizeStations()
{
    if (m_stopping || m_configReply || !isRunning()) return;
    QNetworkRequest request(m_baseUrl.resolved(QUrl(QStringLiteral("/api/local/stations"))));
    request.setRawHeader("X-Local-Secret", m_secret);
    auto *reply = m_network.get(request); m_configReply = reply;
    reply->setReadBufferSize(65536);
    QTimer::singleShot(2000, reply, [reply] { if (!reply->isFinished()) reply->abort(); });
    connect(reply, &QNetworkReply::finished, this, [this, reply] {
        m_configReply = nullptr;
        const auto body = QJsonDocument::fromJson(reply->readAll()).object();
        const bool good = reply->error() == QNetworkReply::NoError
            && reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt() == 200
            && body.value(QStringLiteral("items")).isArray();
        reply->deleteLater();
        if (m_stopping) return;
        if (!good) { emit fault(tr("终端方向配置尚未同步，正在重试")); return; }
        const auto items = body.value(QStringLiteral("items")).toArray();
        if (items.size() != 1) { emit fault(tr("终端配置数量无效")); return; }
        for (const auto &item : items) {
            const auto value = item.toObject();
            const auto id = value.value(QStringLiteral("id")).toString();
            const auto direction = value.value(QStringLiteral("direction")).toString();
            if (id != "board-in" || (direction != "IN" && direction != "OUT")
                || value.value(QStringLiteral("version")).toInt() < 1) {
                emit fault(tr("终端方向配置无效")); return;
            }
        }
        for (const auto &item : items) emit stationConfiguration(item.toObject());
    });
}
