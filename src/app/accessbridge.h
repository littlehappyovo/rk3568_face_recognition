#pragma once
#include <QHash>
#include <QJsonObject>
#include <QLocalServer>
#include <QObject>
#include <QPointer>
#include <QSet>

class FaceBackendService;
class QLocalSocket;

class AccessBridge final : public QObject
{
    Q_OBJECT
public:
    explicit AccessBridge(QByteArray secret, QObject *parent = nullptr);
    ~AccessBridge() override;
    bool listen(const QString &path);
    void attachBackend(FaceBackendService *backend);
    void stop();
public slots:
    void completeRequest(const QString &requestId, const QJsonObject &result, const QJsonObject &error);
signals:
    void commandRequested(const QString &requestId, const QString &operation, const QJsonObject &payload);
    void fault(const QString &message);
private:
    void acceptConnections();
    QByteArray m_secret;
    QLocalServer m_server;
    QHash<QString, QPointer<QLocalSocket>> m_pending;
    QSet<QLocalSocket *> m_connections;
    QString m_path;
};
