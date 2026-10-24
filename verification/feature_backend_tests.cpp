#include "opencvfacebackend.h"
#include "voiceannouncer.h"
#include "mainwindow.h"
#include <QApplication>
#include <QCheckBox>
#include <QDialog>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QListWidget>
#include <QPushButton>
#include <QSignalSpy>
#include <QStandardPaths>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QThread>
#include <QtTest>

class BackendDatabaseTests : public QObject {
    Q_OBJECT
    QString m_root = QString::fromUtf8(FACE_TEST_PROJECT_ROOT);
    void seed(const QString &path) {
        qputenv("FACEAPP_DATA_DIR", path.toUtf8());
        facecore::DatasetStore store((path + "/data").toStdString());
        store.upsertPerson({"P1", "One", facecore::PersonGroup::InGroup, true});
        store.upsertPerson({"P2", "Two", facecore::PersonGroup::OutGroup, false});
        facecore::SampleRecord sample;
        sample.personId = "P1"; sample.sessionId = "SESSION1";
        sample.sampleId = "S1"; sample.preprocessingVersion = "gray128-eq-v1";
        store.saveSample(cv::Mat(128,128,CV_8UC1,cv::Scalar(100)),
                         cv::Mat(112,112,CV_8UC3,cv::Scalar(40,80,120)), sample);
    }
private slots:
    void initTestCase() {
        registerFaceBackendMetaTypes();
        QVERIFY2(QFileInfo::exists(m_root + "/models/face_detection_yunet_2023mar.onnx"),
            "Prepare models/face_detection_yunet_2023mar.onnx before backend tests; see models/README.md");
        qputenv("FACEAPP_CONFIG", (m_root + "/config/rk3568.json").toUtf8());
    }
    void databaseListAndPreviewsUseRealDataset() {
        QTemporaryDir data; seed(data.path());
        OpenCvFaceBackend backend(m_root);
        QSignalSpy people(&backend, &FaceBackendService::faceDatabaseChanged);
        QSignalSpy images(&backend, &FaceBackendService::personSamplesChanged);
        QSignalSpy errors(&backend, &FaceBackendService::errorOccurred);
        backend.refreshFaceDatabase();
        QCOMPARE(people.count(), 1);
        const auto list = people.first()[0].value<FaceDatabasePeople>();
        QCOMPARE(list.size(), 2); QCOMPARE(list[0].sampleCount, 1);
        QVERIFY(list[0].enabled); QVERIFY(!list[1].enabled);
        backend.loadPersonSamples("P1");
        QCOMPARE(images.count(), 1);
        const auto samples = images.last()[1].value<FaceDatabaseSamples>();
        QCOMPARE(samples.size(), 1); QCOMPARE(samples[0].sampleId, QString("S1"));
        QCOMPARE(samples[0].image.pixelColor(20,20), QColor(120,80,40));
        QCOMPARE(errors.count(), 0);
        backend.shutdown();
    }
    void switchingToUnavailableBackendCancelsQueuedSpeech() {
        QTemporaryDir data; seed(data.path());
        OpenCvFaceBackend backend(m_root);
        backend.refreshFaceDatabase();
        VoiceSettings config; config.audioDirectory=data.path();
        config.playerProgram=data.path()+"/player";
        QFile player(config.playerProgram); QVERIFY(player.open(QIODevice::WriteOnly));
        player.write("#!/bin/sh\nsleep 0.15\nexit 0\n"); player.close();
        QVERIFY(player.setPermissions(QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner));
        QFile audio(data.path()+"/morning.wav"); QVERIFY(audio.open(QIODevice::WriteOnly));
        audio.write("test audio"); audio.close();
        backend.m_voice=std::make_unique<VoiceAnnouncer>(config);
        backend.m_recognitionEnabled=true;
        QSignalSpy played(backend.m_voice.get(), &VoiceAnnouncer::playbackRequested);
        FaceObservation first; first.trackId=1; first.personId="P1";
        first.identityState=IdentityState::Known;
        auto second=first; second.trackId=2;
        backend.m_voice->onFrame({first,second},9); QTRY_COMPARE(played.count(),1);
        backend.configureBackend("onnx_embedding");
        QVERIFY(!backend.m_recognitionEnabled);
        QTest::qWait(300); QCOMPARE(played.count(),1);
        backend.shutdown();
    }
    void unavailableAlignedPhotoFallsBackToGray() {
        QTemporaryDir data; seed(data.path());
        facecore::DatasetStore store((data.path() + "/data").toStdString());
        QFile broken(QString::fromStdString((store.root()/store.loadManifest()[0].alignedRelativePath).string()));
        QVERIFY(broken.open(QIODevice::WriteOnly)); broken.close();
        OpenCvFaceBackend backend(m_root);
        QSignalSpy images(&backend, &FaceBackendService::personSamplesChanged);
        backend.loadPersonSamples("P1");
        QCOMPARE(images.count(), 1);
        const auto samples = images.last()[1].value<FaceDatabaseSamples>();
        QVERIFY(!samples[0].image.isNull());
        QCOMPARE(samples[0].image.pixelColor(20,20), QColor(100,100,100));
        backend.shutdown();
    }
    void editingAndDeletionPersistAndDisableRecognition() {
        QTemporaryDir data; seed(data.path());
        OpenCvFaceBackend backend(m_root);
        QSignalSpy errors(&backend, &FaceBackendService::errorOccurred);
        QSignalSpy recognized(&backend, &FaceBackendService::recognitionEnabledChanged);
        backend.updateDatabasePerson({"P1", QString::fromUtf8("新姓名"), "out"}, true);
        facecore::DatasetStore store((data.path() + "/data").toStdString());
        QCOMPARE(store.loadRegistry()[0].displayName, std::string("新姓名"));
        backend.deleteDatabaseSample("P1", "S1");
        QVERIFY(store.loadManifest().empty());
        QVERIFY(!recognized.isEmpty()); QVERIFY(!recognized.last()[0].toBool());
        QCOMPARE(errors.count(), 0);
        backend.deleteDatabasePerson("P1");
        QCOMPARE(store.loadRegistry().size(), std::size_t(1));
        QCOMPARE(store.loadRegistry()[0].personId, std::string("P2"));
        backend.shutdown();
        OpenCvFaceBackend reopened(m_root);
        reopened.configureBackend("opencv_lbph");
        QSignalSpy enabled(&reopened, &FaceBackendService::recognitionEnabledChanged);
        reopened.setRecognitionEnabled(true);
        QCOMPARE(enabled.count(), 1); QVERIFY(!enabled.last()[0].toBool());
        reopened.shutdown();
    }
    void selectedGalleryMustBeRebuiltAfterDeletion() {
        QTemporaryDir data; seed(data.path());
        facecore::DatasetStore store((data.path() + "/data").toStdString());
        auto sample = store.loadManifest()[0]; sample.sampleId = "S2";
        store.saveSample(store.loadGraySample(sample), store.loadAlignedSample(sample), sample);
        OpenCvFaceBackend backend(m_root);
        backend.configureBackend("opencv_lbph");
        QSignalSpy errors(&backend, &FaceBackendService::errorOccurred);
        backend.trainModel();
        QCOMPARE(errors.count(), 0);
        backend.deleteDatabaseSample("P1", "S1");
        QSignalSpy enabled(&backend, &FaceBackendService::recognitionEnabledChanged);
        backend.setRecognitionEnabled(true);
        QVERIFY(!enabled.last()[0].toBool());
        errors.clear();
        backend.trainModel();
        QCOMPARE(errors.count(), 0);
        QVERIFY(!std::filesystem::exists(store.root()/".gallery-stale-opencv_lbph"));
        backend.setRecognitionEnabled(true);
        QVERIFY(enabled.last()[0].toBool());
        backend.shutdown();
        // The old gallery contains the deleted sample. Corrupt the new gallery:
        // restarting must refuse recognition rather than silently load that old data.
        std::filesystem::path newest;
        int galleryCount = 0;
        for (const auto &entry : std::filesystem::directory_iterator((data.path()+"/models").toStdString())) {
            if (entry.path().filename().string().find("lbph_") != 0) continue;
            ++galleryCount;
            if (entry.path() > newest) newest = entry.path();
        }
        QCOMPARE(galleryCount, 2);
        QFile corrupt(QString::fromStdString(newest.string()));
        QVERIFY(corrupt.open(QIODevice::WriteOnly)); corrupt.close();
        OpenCvFaceBackend reopened(m_root);
        reopened.configureBackend("opencv_lbph");
        QSignalSpy afterRestart(&reopened, &FaceBackendService::recognitionEnabledChanged);
        reopened.setRecognitionEnabled(true);
        QVERIFY2(!afterRestart.last()[0].toBool(), "Pre-deletion galleries must not be accepted as fallback");
        reopened.shutdown();
    }

    void realBoardStartupAndDatabase() {
        if (qEnvironmentVariableIntValue("FACEAPP_LIVE_BOARD_TEST") != 1)
            QSKIP("Explicit live-board run required; the camera must be free");
        qunsetenv("FACEAPP_DATA_DIR");
        QCoreApplication::setApplicationName("FaceRecognitionSystem");
        QCoreApplication::setOrganizationName("PortableFaceSystem");
        QFont font("Noto Sans CJK SC"); font.setPixelSize(20); QApplication::setFont(font);
        qApp->setStyleSheet("QLineEdit, QSpinBox, QComboBox { min-height:34px; max-height:42px; }"
            "QPushButton, QToolButton { min-height:42px; max-height:54px; padding:2px 10px; }"
            "QCheckBox { min-height:32px; max-height:40px; }"
            "QProgressBar { min-height:26px; max-height:32px; }");
        auto *backend = new OpenCvFaceBackend(m_root);
        QThread worker;
        backend->moveToThread(&worker);
        connect(&worker, &QThread::finished, backend, &QObject::deleteLater);
        worker.start();
        struct Cleanup {
            OpenCvFaceBackend *backend; QThread &worker;
            ~Cleanup() {
                backend->requestShutdown();
                QMetaObject::invokeMethod(backend, "shutdown", Qt::BlockingQueuedConnection);
                worker.quit(); worker.wait();
            }
        } cleanup{backend, worker};
        MainWindow window(backend);
        window.showFullScreen();
        int frames = 0;
        connect(backend, &FaceBackendService::frameReady, &window, [&frames](const FrameResult &frame) {
            if (!frame.originalImage.isNull()) ++frames;
        });
        QCheckBox *recognition = nullptr;
        for (auto *check : window.findChildren<QCheckBox *>())
            if (check->text() == QString::fromUtf8("开启实时人脸识别")) recognition = check;
        QVERIFY(recognition);
        QTRY_VERIFY_WITH_TIMEOUT(recognition->isChecked(), 15000);
        QTRY_VERIFY_WITH_TIMEOUT(frames >= 10, 15000);
        auto *entry = window.findChild<QPushButton *>("faceDatabaseButton");
        QVERIFY(entry && entry->isEnabled()); entry->click();
        auto *table = window.findChild<QTableWidget *>("databasePeople");
        auto *photos = window.findChild<QListWidget *>("databaseSamples");
        QVERIFY(table && photos);
        QTRY_VERIFY_WITH_TIMEOUT(table->rowCount() > 0, 10000);
        QTRY_VERIFY_WITH_TIMEOUT(photos->count() > 0, 10000);
        qInfo() << "[live-smoke] camera frames=" << frames << "recognition=" << recognition->isChecked()
                << "people=" << table->rowCount() << "first-person samples=" << photos->count()
                << "window=" << window.size();
        const auto screenshot = qgetenv("FACEAPP_TEST_SCREENSHOT_DIR");
        if (!screenshot.isEmpty()) {
            auto *dialog = window.findChild<QDialog *>("faceDatabaseDialog");
            QVERIFY(dialog->grab().save(QString::fromUtf8(screenshot)+"/live-face-database.png"));
        }
        window.findChild<QDialog *>("faceDatabaseDialog")->accept();
    }
};
QTEST_MAIN(BackendDatabaseTests)
#include "feature_backend_tests.moc"
