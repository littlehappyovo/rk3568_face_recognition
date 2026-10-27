#include "accesseventgate.h"
#include "accesspublisher.h"
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QSignalSpy>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QTest>
#include <QUuid>
#include <memory>

class AcknowledgingServer : public QTcpServer {
public:
    explicit AcknowledgingServer(bool correct = true) {
        connect(this, &QTcpServer::newConnection, this, [this, correct] {
            while (hasPendingConnections()) {
                auto *socket = nextPendingConnection();
                auto buffer = std::make_shared<QByteArray>();
                connect(socket, &QTcpSocket::readyRead, socket, [socket, buffer, correct] {
                    buffer->append(socket->readAll());
                    const int split = buffer->indexOf("\r\n\r\n");
                    if (split < 0) return;
                    const auto length = QRegularExpression("Content-Length: ([0-9]+)",
                        QRegularExpression::CaseInsensitiveOption).match(QString::fromLatin1(buffer->left(split)));
                    if (!length.hasMatch() || buffer->size() < split + 4 + length.captured(1).toInt()) return;
                    const auto event = QJsonDocument::fromJson(buffer->mid(split + 4)).object();
                    const QByteArray body = QJsonDocument(QJsonObject{
                        {"event_id", correct ? event.value("event_id") : QJsonValue("wrong-id")},
                        {"persisted", true}, {"duplicate", true}}).toJson(QJsonDocument::Compact);
                    socket->write("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                        + QByteArray::number(body.size()) + "\r\n\r\n" + body);
                    socket->disconnectFromHost();
                });
                connect(socket, &QTcpSocket::disconnected, socket, &QObject::deleteLater);
            }
        });
        if (!listen(QHostAddress::LocalHost, 0)) qFatal("test HTTP listener failed");
    }
    QUrl url() const { return QUrl(QString("http://127.0.0.1:%1").arg(serverPort())); }
};

class AccessEventTests : public QObject {
    Q_OBJECT
    static FaceObservation face(int track = 7) {
        FaceObservation value; value.trackId = track; value.identityState = IdentityState::Known;
        value.personId = "P1"; value.displayName = QString::fromUtf8("测试人员"); return value;
    }
    static QJsonObject event() {
        return {{"event_id", QUuid::createUuid().toString(QUuid::WithoutBraces)},
                {"station_id", "board-in"}, {"direction_version", 1},
                {"stream_id", QUuid::createUuid().toString(QUuid::WithoutBraces)},
                {"track_id", 7}, {"observed_at", "2026-10-26T01:00:00.000Z"},
                {"person_id", "P1"}, {"display_name", "Test"}, {"decision", "allowed"}};
    }
private slots:
    void staticImagesNeverCreateAccessEvents() {
        AccessEventGate gate;
        const auto stream = QUuid::createUuid().toString(QUuid::WithoutBraces);
        const auto now = QDateTime::currentDateTimeUtc();
        QVERIFY(gate.observe("board-in", stream, 1, {face()}, {7}, now, false).isEmpty());
        QCOMPARE(gate.observe("board-in", stream, 1, {face()}, {7}, now, true).size(), 1);
    }
    void stableTrackWritesOnceUntilConfirmedLoss() {
        AccessEventGate gate;
        const QDateTime now = QDateTime::fromString("2026-10-26T01:00:00Z", Qt::ISODate);
        const QString stream = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QCOMPARE(gate.observe("board-in", stream, 1, {face()}, {7}, now).size(), 1);
        QVERIFY(gate.observe("board-in", stream, 1, {}, {7}, now).isEmpty());
        QVERIFY(gate.observe("board-in", stream, 1, {face()}, {7}, now).isEmpty());
        QVERIFY(gate.observe("board-in", stream, 1, {}, {}, now).isEmpty());
        QCOMPARE(gate.observe("board-in", stream, 1, {face()}, {7}, now).size(), 1);
    }
    void restartAndDirectionChangeDoNotReuseDedupState() {
        AccessEventGate gate; const auto now = QDateTime::currentDateTimeUtc();
        const QString first = QUuid::createUuid().toString(QUuid::WithoutBraces);
        const QString second = QUuid::createUuid().toString(QUuid::WithoutBraces);
        const auto before = gate.observe("board-in", first, 1, {face()}, {7}, now);
        const auto after = gate.observe("board-in", second, 2, {face()}, {7}, now);
        QCOMPARE(after.size(), 1); QVERIFY(before[0]["event_id"] != after[0]["event_id"]);
        QCOMPARE(after[0]["direction_version"].toInt(), 2);
    }
    void unconfirmedAndUnknownNeverAcquireInventedIdentity() {
        AccessEventGate gate; const auto now = QDateTime::currentDateTimeUtc();
        const QString stream = QUuid::createUuid().toString(QUuid::WithoutBraces);
        auto value = face(); value.identityState = IdentityState::DetectionOnly;
        QVERIFY(gate.observe("board-in", stream, 1, {value}, {7}, now).isEmpty());
        value.identityState = IdentityState::Unknown;
        const auto denied = gate.observe("board-in", stream, 1, {value}, {7}, now);
        QCOMPARE(denied.size(), 1); QVERIFY(denied[0]["person_id"].isNull());
        QCOMPARE(denied[0]["decision"].toString(), QString("denied"));
        QVERIFY(gate.observe("board-in", stream, 0, {face(8)}, {8}, now).isEmpty());
    }
    void unknownThenKnownOnSameTrackRecordsBothDecisionsOnce() {
        AccessEventGate gate; const auto now = QDateTime::currentDateTimeUtc();
        const QString stream = QUuid::createUuid().toString(QUuid::WithoutBraces);
        auto unknown = face(); unknown.identityState = IdentityState::Unknown;
        const auto denied = gate.observe("board-in", stream, 1, {unknown}, {7}, now);
        QCOMPARE(denied.size(), 1);
        QCOMPARE(denied[0]["decision"].toString(), QString("denied"));
        const auto allowed = gate.observe("board-in", stream, 1, {face()}, {7}, now.addSecs(1));
        QCOMPARE(allowed.size(), 1);
        QCOMPARE(allowed[0]["decision"].toString(), QString("allowed"));
        QCOMPARE(allowed[0]["person_id"].toString(), QString("P1"));
        QVERIFY(denied[0]["event_id"].toString() != allowed[0]["event_id"].toString());
        QVERIFY(gate.observe("board-in", stream, 1, {face()}, {7}, now.addSecs(2)).isEmpty());
        auto laterUnknown = face(); laterUnknown.identityState = IdentityState::Unknown;
        QVERIFY(gate.observe("board-in", stream, 1, {laterUnknown}, {7}, now.addSecs(3)).isEmpty());
    }
    void outboxSurvivesTimeoutRestartAndDuplicateAck() {
        QTemporaryDir directory;
        QTcpServer unused; QVERIFY(unused.listen(QHostAddress::LocalHost, 0));
        const auto url = QUrl(QString("http://127.0.0.1:%1").arg(unused.serverPort())); unused.close();
        {
            AccessPublisher publisher(directory.path()); QSignalSpy faults(&publisher, &AccessPublisher::fault);
            publisher.setEndpoint(url, "test-startup-secret"); publisher.enqueue(event());
            QCOMPARE(publisher.pendingCount(), 1); QTRY_VERIFY(!faults.isEmpty());
        }
        AcknowledgingServer server;
        AccessPublisher restarted(directory.path()); QCOMPARE(restarted.pendingCount(), 1);
        restarted.setEndpoint(server.url(), "test-startup-secret");
        QTRY_COMPARE(restarted.pendingCount(), 0);
    }
    void mismatchedAckMustNotDeletePendingEvent() {
        QTemporaryDir directory; AcknowledgingServer server(false);
        AccessPublisher publisher(directory.path()); QSignalSpy faults(&publisher, &AccessPublisher::fault);
        publisher.setEndpoint(server.url(), "test-startup-secret"); publisher.enqueue(event());
        QTRY_VERIFY(!faults.isEmpty()); QCOMPARE(publisher.pendingCount(), 1);
    }
    void diskWriteFailureReportsFault() {
        QTemporaryDir directory; QFile blocker(directory.filePath("file"));
        QVERIFY(blocker.open(QIODevice::WriteOnly)); blocker.close();
        AccessPublisher publisher(directory.filePath("file/queue"));
        QSignalSpy faults(&publisher, &AccessPublisher::fault);
        publisher.enqueue(event()); QVERIFY(!faults.isEmpty()); QCOMPARE(publisher.pendingCount(), 0);
    }
};
QTEST_GUILESS_MAIN(AccessEventTests)
#include "access_event_tests.moc"
