#include "accessbridge.h"
#include "facebackendservice.h"
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QLocalSocket>
#include <QTimer>
#include <QUuid>
#include <memory>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr quint32 maxMessage = 12*1024*1024;
QByteArray encode(const QJsonObject &value) {
    const auto data = QJsonDocument(value).toJson(QJsonDocument::Compact);
    QByteArray header(4, '\0');
    for (int index=0; index<4; ++index) header[index] = char(quint32(data.size()) >> (24-index*8));
    return header + data;
}
bool sameSecret(const QByteArray &first, const QByteArray &second) {
    if (first.size() != second.size()) return false;
    unsigned char difference = 0;
    for (int index=0; index<first.size(); ++index) difference |= first[index] ^ second[index];
    return difference == 0;
}
}

AccessBridge::AccessBridge(QByteArray secret, QObject *parent)
    : QObject(parent), m_secret(std::move(secret))
{
    m_server.setSocketOptions(QLocalServer::UserAccessOption);
    m_server.setMaxPendingConnections(8);
    connect(&m_server, &QLocalServer::newConnection, this, &AccessBridge::acceptConnections);
}
AccessBridge::~AccessBridge() { stop(); }

bool AccessBridge::listen(const QString &path)
{
    const QString directory = QFileInfo(path).absolutePath();
    if (!QDir().mkpath(directory) || !QFile::setPermissions(directory,
        QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner)) return false;
    struct stat info{};
    const auto native = QFile::encodeName(path);
    if (::lstat(native.constData(), &info) == 0) {
        QLocalSocket existing; existing.connectToServer(path);
        if (existing.waitForConnected(100) || !S_ISSOCK(info.st_mode) || info.st_uid != ::getuid()) {
            emit fault(tr("管理控制套接字已被占用")); return false;
        }
        QLocalServer::removeServer(path);
    }
    if (!m_server.listen(path)) { emit fault(tr("无法建立本机管理控制通道")); return false; }
    m_path = path; return true;
}

void AccessBridge::attachBackend(FaceBackendService *backend)
{
    connect(this, &AccessBridge::commandRequested, backend, &FaceBackendService::executeAccessCommand, Qt::QueuedConnection);
    connect(backend, &FaceBackendService::accessCommandFinished, this, &AccessBridge::completeRequest, Qt::QueuedConnection);
}

void AccessBridge::acceptConnections()
{
    const QSet<QString> operations{"people.list", "people.create", "people.update", "people.delete",
        "samples.list", "samples.read", "samples.add", "samples.delete", "gallery.rebuild", "station.configure"};
    while (m_server.hasPendingConnections()) {
        auto *socket = m_server.nextPendingConnection();
        if (m_connections.size() >= 8) { socket->abort(); socket->deleteLater(); continue; }
        m_connections.insert(socket); socket->setReadBufferSize(maxMessage+4);
        auto buffer = std::make_shared<QByteArray>();
        connect(socket, &QLocalSocket::readyRead, this, [this, socket, buffer, operations] {
            buffer->append(socket->readAll());
            if (buffer->size() < 4) return;
            quint32 size = 0;
            for (int index=0; index<4; ++index) size = (size << 8) | static_cast<unsigned char>((*buffer)[index]);
            if (size == 0 || size > maxMessage || buffer->size() > int(size+4)) { socket->abort(); return; }
            if (buffer->size() < int(size+4)) return;
            QJsonParseError parse;
            const auto request = QJsonDocument::fromJson(buffer->mid(4), &parse).object();
            const QString id = request.value(QStringLiteral("request_id")).toString();
            const QString operation = request.value(QStringLiteral("operation")).toString();
            if (parse.error != QJsonParseError::NoError || QUuid(id).isNull() || m_pending.contains(id)
                || !operations.contains(operation) || !request.value(QStringLiteral("payload")).isObject()
                || !sameSecret(request.value(QStringLiteral("secret")).toString().toUtf8(), m_secret)) {
                socket->abort(); return;
            }
            // One request per socket; forbid pipelining and give queued frames an expiry.
            disconnect(socket, &QLocalSocket::readyRead, this, nullptr);
            m_pending.insert(id, socket);
            auto payload = request.value(QStringLiteral("payload")).toObject();
            emit commandRequested(id, operation, payload);
        });
        connect(socket, &QLocalSocket::disconnected, this, [this, socket] {
            for (auto iterator=m_pending.begin(); iterator!=m_pending.end();) {
                if (iterator.value() == socket) iterator=m_pending.erase(iterator); else ++iterator;
            }
            m_connections.remove(socket); socket->deleteLater();
        });
        QTimer::singleShot(190000, socket, [socket] { socket->abort(); });
    }
}

void AccessBridge::completeRequest(const QString &id, const QJsonObject &result, const QJsonObject &error)
{
    const auto socket = m_pending.take(id);
    if (!socket) return;
    const auto data = encode({{"request_id", id}, {"ok", error.isEmpty()},
        {"result", result}, {"error", error.isEmpty() ? QJsonValue(QJsonValue::Null) : QJsonValue(error)}});
    if (data.size() > int(maxMessage+4)) { socket->abort(); return; }
    socket->write(data); socket->disconnectFromServer();
}

void AccessBridge::stop()
{
    m_server.close();
    const auto sockets = m_connections;
    for (auto *socket : sockets) { disconnect(socket, nullptr, this, nullptr); socket->abort(); socket->deleteLater(); }
    m_connections.clear(); m_pending.clear();
    if (!m_path.isEmpty()) { QLocalServer::removeServer(m_path); m_path.clear(); }
}
