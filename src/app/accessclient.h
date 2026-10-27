#pragma once
#include <QDateTime>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QObject>
#include <QSet>
#include <QUrl>
#include <functional>

class QNetworkReply;

class AccessClient final : public QObject
{
    Q_OBJECT
public:
    explicit AccessClient(QObject *parent = nullptr);
    ~AccessClient() override;
    void setEndpoint(const QUrl &baseUrl, const QByteArray &localSecret);
    QString sessionId() const { return m_sessionId; }
    QDateTime expiresAt() const { return m_expiresAt; }
    void authorizeAction(std::function<void()> action);
public slots:
    void authStatus();
    void setupPassword(const QString &password);
    void login(const QString &password);
    void refreshSession();
    void checkSession();
    void logout();
    void updateDirection(const QString &direction);
signals:
    void authStatusReady(bool initialized);
    void authenticated(const QString &sessionId, const QDateTime &expiresAt);
    void sessionRefreshed(const QDateTime &expiresAt);
    void sessionLost();
    void errorOccurred(const QString &message);
private:
    using Success = std::function<void(const QJsonObject &)>;
    void send(const QByteArray &method, const QString &path, const QJsonObject &body,
              Success success, bool protectedRequest = false, bool quiet = false);
    void cancelRequests();
    void invalidate();
    void updateExpiry(const QJsonObject &body);
    QNetworkAccessManager m_network;
    QSet<QNetworkReply *> m_requests;
    QUrl m_baseUrl;
    QByteArray m_localSecret, m_csrf;
    QString m_sessionId;
    QDateTime m_expiresAt;
    quint64 m_generation = 0;
};
