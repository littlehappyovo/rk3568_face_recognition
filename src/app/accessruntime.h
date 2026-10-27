#pragma once
#include <QByteArray>
#include <QObject>
#include <QProcess>
#include <QTimer>
#include <QUrl>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QPointer>
#include <memory>

class QLockFile;
class QNetworkReply;

class AccessRuntime final : public QObject
{
    Q_OBJECT
public:
    explicit AccessRuntime(QString applicationDirectory, QObject *parent = nullptr);
    ~AccessRuntime() override;
    void start(const QString &runtimeDirectory, const QString &pythonExecutable = {});
    void stop();
    bool isRunning() const;
    QByteArray localSecret() const { return m_secret; }
signals:
    void ready(const QUrl &baseUrl);
    void fault(const QString &message);
    void stationConfiguration(const QJsonObject &configuration);
private:
    void launch();
    void scheduleRestart();
    void synchronizeStations();
    QString m_applicationDirectory, m_directory, m_python;
    QByteArray m_secret, m_output;
    QProcess m_process;
    QTimer m_restart;
    QTimer m_configTimer;
    QNetworkAccessManager m_network;
    QPointer<QNetworkReply> m_configReply;
    QUrl m_baseUrl;
    std::unique_ptr<QLockFile> m_lock;
    bool m_stopping = true;
    int m_backoff = 1000;
    int m_port = 8080;
};
