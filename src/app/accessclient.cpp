#include "accessclient.h"
#include <QJsonDocument>
#include <QNetworkCookie>
#include <QNetworkCookieJar>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QTimer>
#include <QUuid>

AccessClient::AccessClient(QObject *parent) : QObject(parent) {}
AccessClient::~AccessClient() { cancelRequests(); }

void AccessClient::cancelRequests()
{
    ++m_generation;
    const auto pending = m_requests; m_requests.clear();
    for (auto *reply : pending) { disconnect(reply,nullptr,this,nullptr); reply->abort(); reply->deleteLater(); }
}

void AccessClient::invalidate()
{
    const bool hadSession = !m_sessionId.isEmpty();
    m_sessionId.clear(); m_csrf.clear(); m_expiresAt = {};
    cancelRequests();
    for (const auto &cookie : m_network.cookieJar()->cookiesForUrl(m_baseUrl))
        m_network.cookieJar()->deleteCookie(cookie);
    if (hadSession) emit sessionLost();
}

void AccessClient::setEndpoint(const QUrl &baseUrl, const QByteArray &localSecret)
{
    invalidate();
    if (baseUrl.scheme() != QStringLiteral("http")
        || (baseUrl.host() != QStringLiteral("127.0.0.1") && baseUrl.host() != QStringLiteral("::1"))) {
        m_baseUrl = QUrl(); emit errorOccurred(tr("管理认证只连接板端本机服务")); return;
    }
    m_baseUrl = baseUrl; m_localSecret = localSecret;
}

void AccessClient::send(const QByteArray &method, const QString &path, const QJsonObject &body,
                        Success success, bool protectedRequest, bool quiet)
{
    if (!m_baseUrl.isValid() || m_baseUrl.isEmpty() || m_requests.size() >= 8) {
        if (!quiet) emit errorOccurred(tr("管理服务尚未连接或请求繁忙"));
        if (protectedRequest) invalidate();
        return;
    }
    QNetworkRequest request(m_baseUrl.resolved(QUrl(path)));
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    request.setRawHeader("Origin", m_baseUrl.adjusted(QUrl::RemovePath | QUrl::RemoveQuery | QUrl::RemoveFragment).toEncoded());
    if (!m_csrf.isEmpty()) request.setRawHeader("X-CSRF-Token", m_csrf);
    if (path == QStringLiteral("/api/local/setup")) request.setRawHeader("X-Local-Secret", m_localSecret);
    const QByteArray data = QJsonDocument(body).toJson(QJsonDocument::Compact);
    QNetworkReply *reply = method == "GET" ? m_network.get(request)
        : m_network.sendCustomRequest(request, method, data);
    m_requests.insert(reply); reply->setReadBufferSize(65536);
    const auto generation = m_generation;
    QTimer::singleShot(2000, reply, [reply] { if (!reply->isFinished()) reply->abort(); });
    connect(reply, &QNetworkReply::finished, this,
        [this, reply, generation, success=std::move(success), protectedRequest, quiet] {
        m_requests.remove(reply);
        const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        QJsonParseError parse;
        const auto document = QJsonDocument::fromJson(reply->readAll(), &parse);
        const bool good = reply->error() == QNetworkReply::NoError && status >= 200 && status < 300
            && parse.error == QJsonParseError::NoError && document.isObject();
        reply->deleteLater();
        if (generation != m_generation) return;
        if (!good) {
            const auto error = document.object().value(QStringLiteral("error")).toObject();
            QString message = error.value(QStringLiteral("message")).toString();
            if (message.isEmpty()) message = tr("管理服务未响应，请稍后重试");
            if (protectedRequest) invalidate();
            if (!quiet) emit errorOccurred(message);
            return;
        }
        success(document.object());
    });
}

void AccessClient::authStatus()
{
    send("GET", QStringLiteral("/api/auth/status"), {}, [this](const QJsonObject &body) {
        emit authStatusReady(body.value(QStringLiteral("initialized")).toBool());
    });
}

void AccessClient::setupPassword(const QString &password)
{
    send("POST", QStringLiteral("/api/local/setup"), {{"password",password}},
        [this, password](const QJsonObject &) { login(password); });
}

void AccessClient::updateExpiry(const QJsonObject &body)
{
    m_expiresAt = QDateTime::fromMSecsSinceEpoch(qint64(body.value(QStringLiteral("expires_at")).toDouble()*1000), Qt::UTC);
    emit sessionRefreshed(m_expiresAt);
}

void AccessClient::login(const QString &password)
{
    send("POST", QStringLiteral("/api/auth/login"), {{"password",password}}, [this](const QJsonObject &body) {
        const auto csrf = body.value(QStringLiteral("csrf_token")).toString().toLatin1();
        if (csrf.isEmpty() || csrf.size() > 128 || body.value(QStringLiteral("expires_at")).toDouble() <= 0) {
            invalidate(); emit errorOccurred(tr("管理服务返回的会话无效")); return;
        }
        m_csrf = csrf; m_sessionId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        updateExpiry(body); emit authenticated(m_sessionId, m_expiresAt);
    });
}

void AccessClient::refreshSession()
{
    if (m_sessionId.isEmpty()) return;
    send("GET", QStringLiteral("/api/auth/session"), {}, [this](const QJsonObject &body) { updateExpiry(body); }, true);
}

void AccessClient::checkSession()
{
    if (m_sessionId.isEmpty()) return;
    send("GET", QStringLiteral("/api/auth/check"), {}, [this](const QJsonObject &body) { updateExpiry(body); }, true);
}

void AccessClient::authorizeAction(std::function<void()> action)
{
    if (m_sessionId.isEmpty()) { emit sessionLost(); return; }
    send("GET", QStringLiteral("/api/auth/check"), {}, [this, action=std::move(action)](const QJsonObject &body) {
        updateExpiry(body); if (!m_sessionId.isEmpty()) action();
    }, true);
}

void AccessClient::logout()
{
    if (m_sessionId.isEmpty()) { cancelRequests(); return; }
    cancelRequests();
    // Qt attaches the current cookie when creating this request, then local state is cleared.
    send("POST", QStringLiteral("/api/auth/logout"), {}, [](const QJsonObject &) {}, false, true);
    m_sessionId.clear(); m_csrf.clear(); m_expiresAt = {};
    for (const auto &cookie : m_network.cookieJar()->cookiesForUrl(m_baseUrl))
        m_network.cookieJar()->deleteCookie(cookie);
    emit sessionLost();
}

void AccessClient::updateDirection(const QString &direction)
{
    send("PATCH", QStringLiteral("/api/stations/board-in"), {{"direction",direction}}, [](const QJsonObject &) {}, true);
}
