#include "accesspublisher.h"
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSaveFile>
#include <QUuid>
#include <algorithm>

AccessPublisher::AccessPublisher(QString directory, QObject *parent)
    : QObject(parent), m_directory(std::move(directory))
{
    m_timeout.setSingleShot(true);
    m_retry.setSingleShot(true);
    connect(&m_timeout, &QTimer::timeout, this, [this] { if (m_reply) m_reply->abort(); });
    connect(&m_retry, &QTimer::timeout, this, &AccessPublisher::flush);
}

AccessPublisher::~AccessPublisher()
{
    m_timeout.stop(); m_retry.stop();
    if (m_reply) {
        disconnect(m_reply, nullptr, this, nullptr);
        m_reply->abort();
    }
}

QStringList AccessPublisher::pendingFiles() const
{
    return QDir(m_directory).entryList({QStringLiteral("*.json")}, QDir::Files, QDir::Name);
}

int AccessPublisher::pendingCount() const { return pendingFiles().size(); }

void AccessPublisher::setEndpoint(const QUrl &baseUrl, const QByteArray &secret)
{
    m_baseUrl = baseUrl; m_secret = secret; m_backoffMs = 1000;
    QTimer::singleShot(0, this, &AccessPublisher::flush);
}

void AccessPublisher::enqueue(const QJsonObject &event)
{
    const QString id = event.value(QStringLiteral("event_id")).toString();
    const QDateTime observed = QDateTime::fromString(event.value(QStringLiteral("observed_at")).toString(), Qt::ISODateWithMs);
    QDir directory(m_directory);
    if (QUuid(id).isNull() || !observed.isValid() || !directory.mkpath(QStringLiteral("."))) {
        emit fault(tr("识别记录无法写入待发目录，请检查磁盘与运行配置")); return;
    }
    const QString name = observed.toUTC().toString(QStringLiteral("yyyyMMddHHmmsszzz_")) + id + QStringLiteral(".json");
    const QString path = directory.filePath(name);
    const QByteArray data = QJsonDocument(event).toJson(QJsonDocument::Compact);
    if (QFile::exists(path)) {
        QFile old(path);
        if (!old.open(QIODevice::ReadOnly) || old.readAll() != data) {
            emit fault(tr("待发记录ID冲突，原始记录已保留")); return;
        }
    } else {
        QSaveFile file(path);
        if (!file.open(QIODevice::WriteOnly)
            || !file.setPermissions(QFile::ReadOwner | QFile::WriteOwner)
            || file.write(data) != data.size() || !file.commit()) {
            emit fault(tr("识别记录写入失败，请检查剩余空间；该次记录未持久化")); return;
        }
    }
    emit pendingChanged(pendingCount());
    flush();
}

void AccessPublisher::retry(const QString &message)
{
    emit fault(message);
    m_retry.start(m_backoffMs);
    m_backoffMs = std::min(30000, m_backoffMs * 2);
}

void AccessPublisher::flush()
{
    if (m_reply || !m_baseUrl.isValid() || m_baseUrl.isEmpty() || m_secret.isEmpty()) return;
    const auto files = pendingFiles();
    if (files.isEmpty()) return;
    const QString path = QDir(m_directory).filePath(files.front());
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly) || file.size() > 65536) {
        retry(tr("待发识别记录不可读，已保留原文件")); return;
    }
    const QByteArray data = file.readAll();
    QJsonParseError parseError;
    const auto document = QJsonDocument::fromJson(data, &parseError);
    const QString id = document.object().value(QStringLiteral("event_id")).toString();
    if (parseError.error != QJsonParseError::NoError || QUuid(id).isNull()) {
        retry(tr("待发识别记录损坏，已保留原文件")); return;
    }
    QNetworkRequest request(m_baseUrl.resolved(QUrl(QStringLiteral("/api/local/events"))));
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    request.setRawHeader("X-Local-Secret", m_secret);
    auto *reply = m_network.post(request, data);
    m_reply = reply; m_timeout.start(2000);
    connect(reply, &QNetworkReply::finished, this, [this, reply, path, id] {
        m_timeout.stop(); m_reply = nullptr;
        const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        const auto ack = QJsonDocument::fromJson(reply->readAll()).object();
        const bool persisted = reply->error() == QNetworkReply::NoError && status == 200
            && ack.value(QStringLiteral("persisted")).toBool()
            && ack.value(QStringLiteral("event_id")).toString() == id;
        reply->deleteLater();
        if (!persisted) {
            retry(tr("识别记录尚未获后台确认，已保留并等待重试")); return;
        }
        if (!QFile::remove(path)) {
            retry(tr("记录已入库，但待发文件无法清理；重试不会重复入库")); return;
        }
        m_backoffMs = 1000; emit pendingChanged(pendingCount());
        QTimer::singleShot(0, this, &AccessPublisher::flush);
    });
}
