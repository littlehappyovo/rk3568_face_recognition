#pragma once

#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QObject>
#include <QPointer>
#include <QTimer>
#include <QUrl>

class QNetworkReply;

class AccessPublisher final : public QObject
{
    Q_OBJECT
public:
    explicit AccessPublisher(QString directory, QObject *parent = nullptr);
    ~AccessPublisher() override;
    int pendingCount() const;
    void setEndpoint(const QUrl &baseUrl, const QByteArray &secret);
public slots:
    void enqueue(const QJsonObject &event);
    void flush();
signals:
    void pendingChanged(int count);
    void fault(const QString &message);
private:
    QStringList pendingFiles() const;
    void retry(const QString &message);
    QString m_directory;
    QUrl m_baseUrl;
    QByteArray m_secret;
    QNetworkAccessManager m_network;
    QPointer<QNetworkReply> m_reply;
    QTimer m_timeout;
    QTimer m_retry;
    int m_backoffMs = 1000;
};
