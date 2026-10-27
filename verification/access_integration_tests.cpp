#include "accessbridge.h"
#include "accessruntime.h"
#include "accesspublisher.h"
#include "opencvfacebackend.h"
#include <facecore/dataset.hpp>
#include <QJsonDocument>
#include <QJsonArray>
#include <QLocalSocket>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>
#include <QUuid>

static QByteArray packet(const QJsonObject &request) {
    const auto data = QJsonDocument(request).toJson(QJsonDocument::Compact);
    QByteArray result(4, '\0');
    const auto size = static_cast<quint32>(data.size());
    for (int index=0; index<4; ++index) result[index] = char(size >> (24 - 8*index));
    return result + data;
}

class AccessIntegrationTests : public QObject {
    Q_OBJECT
private slots:
    void initTestCase() {
        registerFaceBackendMetaTypes();
        qputenv("FACEAPP_CONFIG", QByteArray(FACE_TEST_PROJECT_ROOT) + "/config/rk3568.json");
    }
    void cleanup() { qunsetenv("FACEAPP_DATA_DIR"); qunsetenv("FACEAPP_ACCESS_PORT"); }
    void runtimeStartsPrivateServerAndStops() {
        QTemporaryDir directory; qputenv("FACEAPP_ACCESS_PORT", "0");
        AccessRuntime runtime(QString::fromUtf8(FACE_TEST_PROJECT_ROOT));
        QSignalSpy ready(&runtime, &AccessRuntime::ready);
        QSignalSpy configurations(&runtime, &AccessRuntime::stationConfiguration);
        QSignalSpy faults(&runtime, &AccessRuntime::fault);
        runtime.start(directory.path(), QString::fromUtf8(FACE_TEST_ACCESS_PYTHON));
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 5000);
        QVERIFY(ready[0][0].toUrl().port() > 0); QCOMPARE(faults.count(), 0);
        QTRY_COMPARE_WITH_TIMEOUT(configurations.count(), 1, 3000);
        QCOMPARE(configurations[0][0].toJsonObject().value("id").toString(), QString("board-in"));
        QCOMPARE(configurations[0][0].toJsonObject().value("version").toInt(), 1);
        runtime.stop(); QVERIFY(!runtime.isRunning());
        QVERIFY(QFileInfo::exists(directory.filePath("access.sqlite3")));
    }
    void ipcRejectsWrongSecretOversizedAndMismatchedReply() {
        QTemporaryDir directory; AccessBridge bridge("test-startup-secret");
        const QString path = directory.filePath("control.sock"); QVERIFY(bridge.listen(path));
        for (bool oversized : {false, true}) {
            QLocalSocket client; client.connectToServer(path);
            QTRY_COMPARE(client.state(), QLocalSocket::ConnectedState);
            client.write(oversized ? QByteArray::fromHex("00d00000") : packet({
                {"request_id", QUuid::createUuid().toString(QUuid::WithoutBraces)},
                {"operation", "people.list"}, {"secret", "wrong-secret"}, {"payload", QJsonObject{}}}));
            QTRY_COMPARE(client.state(), QLocalSocket::UnconnectedState);
        }
        QLocalSocket valid; valid.connectToServer(path);
        QTRY_COMPARE(valid.state(), QLocalSocket::ConnectedState);
        const QString id = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QSignalSpy requested(&bridge, &AccessBridge::commandRequested);
        valid.write(packet({{"request_id", id}, {"operation", "people.list"},
                            {"secret", "test-startup-secret"}, {"payload", QJsonObject{}}}));
        QTRY_COMPARE(requested.count(), 1);
        bridge.completeRequest("unrelated-id", {{"items", QJsonArray{}}}, {});
        QCOMPARE(valid.bytesAvailable(), qint64(0));
        bridge.completeRequest(id, {{"items", QJsonArray{}}}, {});
        QTRY_VERIFY(valid.bytesAvailable() > 4);
        const auto response = QJsonDocument::fromJson(valid.readAll().mid(4)).object();
        QCOMPARE(response["request_id"].toString(), id); QVERIFY(response["ok"].toBool());
        bridge.stop();
    }
    void ipcRejectsRemoteFrameCommands() {
        QTemporaryDir directory; AccessBridge bridge("test-startup-secret");
        const QString path = directory.filePath("control.sock"); QVERIFY(bridge.listen(path));
        for (const QString operation : {QString("frame.submit"), QString("frame.tracks")}) {
            QLocalSocket client; client.connectToServer(path);
            QTRY_COMPARE(client.state(), QLocalSocket::ConnectedState);
            QSignalSpy requested(&bridge, &AccessBridge::commandRequested);
            client.write(packet({{"request_id", QUuid::createUuid().toString(QUuid::WithoutBraces)},
                {"operation", operation}, {"secret", "test-startup-secret"}, {"payload", QJsonObject{}}}));
            QTest::qWait(100);
            QCOMPARE(requested.count(), 0);
            QCOMPARE(client.state(), QLocalSocket::UnconnectedState);
        }
        bridge.stop();
    }
    void webAndLocalDatabaseChangesSerializeAndInvalidateGallery() {
        QTemporaryDir directory; qputenv("FACEAPP_DATA_DIR", directory.path().toUtf8());
        facecore::DatasetStore store((directory.path()+"/data").toStdString());
        store.upsertPerson({"P1", "Original", facecore::PersonGroup::InGroup, true});
        facecore::SampleRecord sample;
        sample.personId="P1"; sample.sessionId="SESSION1"; sample.sampleId="S1";
        sample.preprocessingVersion="gray128-eq-v1";
        store.saveSample(cv::Mat(128,128,CV_8UC1,cv::Scalar(100)),
            cv::Mat(112,112,CV_8UC3,cv::Scalar(40,80,120)), sample);
        OpenCvFaceBackend backend(QString::fromUtf8(FACE_TEST_PROJECT_ROOT));
        backend.configureBackend("opencv_lbph"); backend.trainModel();
        QSignalSpy replies(&backend, &FaceBackendService::accessCommandFinished);
        backend.executeAccessCommand("web-update", "people.update", {{"person_id", "P1"},
            {"display_name", "Renamed"}, {"group", "out_group"}, {"enabled", true}});
        QCOMPARE(replies.count(), 1); QCOMPARE(replies[0][0].toString(), QString("web-update"));
        QVERIFY(replies[0][2].toJsonObject().isEmpty());
        QCOMPARE(store.loadRegistry()[0].displayName, std::string("Renamed"));
        backend.deleteDatabaseSample("P1", "S1");
        backend.executeAccessCommand("web-delete-again", "samples.delete", {{"person_id", "P1"}, {"sample_id", "S1"}});
        QCOMPARE(replies.count(), 2); QVERIFY(!replies[1][2].toJsonObject().isEmpty());
        QVERIFY(std::filesystem::exists(store.root()/".gallery-stale-opencv_lbph"));
        QVERIFY(store.loadManifest().empty()); backend.shutdown();
    }
};
QTEST_GUILESS_MAIN(AccessIntegrationTests)
#include "access_integration_tests.moc"
