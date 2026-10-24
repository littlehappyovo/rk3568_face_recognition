#include "mainwindow.h"
#include "mockfacebackend.h"
#include "facedatabasedialog.h"
#include <QCheckBox>
#include <QDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QSignalSpy>
#include <QTimer>
#include <QTableWidget>
#include <QListWidget>
#include <QPainter>
#include <QDir>
#include <QAbstractButton>
#include <QMessageBox>
#include <QtTest>

class FeatureUiTests : public QObject {
    Q_OBJECT
    static void populateDatabase(MockFaceBackend &backend) {
        backend.faceDatabaseChanged({{{"P1", "One", "in"}, true, 1},
                                     {{"P2", "Two", "out"}, true, 1}});
        QCoreApplication::processEvents();
        FaceDatabaseSample sample;
        sample.sampleId = "S1";
        backend.personSamplesChanged("P1", {sample});
    }

    static QPushButton *deleteButton(FaceDatabaseDialog &dialog, const QString &text) {
        for (auto *button : dialog.findChildren<QPushButton *>())
            if (button->text() == text) return button;
        return nullptr;
    }
private slots:
    void initTestCase() { registerFaceBackendMetaTypes(); }

    void captureRequestsTenSamplesByDefault() {
        MockFaceBackend backend;
        MainWindow window(&backend);
        window.findChild<QLineEdit *>("capturePersonId")->setText("P001");
        window.findChild<QLineEdit *>("capturePersonName")->setText(QString::fromUtf8("测试人员"));
        QSignalSpy capture(&window, &MainWindow::beginCaptureRequested);
        QVERIFY(QMetaObject::invokeMethod(&window, "beginCapture"));
        QCOMPARE(capture.count(), 1);
        QCOMPARE(capture.first().at(1).toInt(), 10);
        backend.shutdown();
    }

    void launchStartsCameraAndRecognition() {
        MockFaceBackend backend;
        MainWindow window(&backend);
        QSignalSpy states(&backend, &FaceBackendService::stateChanged);
        QCheckBox *recognition = nullptr;
        for (auto *check : window.findChildren<QCheckBox *>())
            if (check->text() == QString::fromUtf8("开启实时人脸识别")) recognition = check;
        QVERIFY(recognition);
        QTRY_VERIFY_WITH_TIMEOUT(recognition->isChecked(), 1500);
        bool streaming = false;
        for (const auto &state : states)
            streaming |= state[0].value<ServiceState>() == ServiceState::Streaming;
        QVERIFY(streaming);
        backend.shutdown();
    }

    void databaseManagerOpensFromMainWindow() {
        MockFaceBackend backend;
        MainWindow window(&backend);
        auto *button = window.findChild<QPushButton *>("faceDatabaseButton");
        QVERIFY2(button, "The face database must have a visible management entry");
        window.show();
        button->click();
        auto *dialog = window.findChild<QDialog *>("faceDatabaseDialog");
        QTRY_VERIFY(dialog && dialog->isVisible());
        dialog->accept();
        backend.shutdown();
    }

    void maintenanceLaunchLeavesCameraStopped() {
        MockFaceBackend backend;
        MainWindow window(&backend, false);
        QSignalSpy states(&backend, &FaceBackendService::stateChanged);
        QTest::qWait(20);
        QVERIFY(!states.isEmpty());
        for (const auto &state : states)
            QVERIFY(state[0].value<ServiceState>() != ServiceState::Streaming);
        backend.shutdown();
    }

    void personDeletionKeepsConfirmedIdAcrossRegistryRefresh() {
        MockFaceBackend backend;
        FaceDatabaseDialog dialog(&backend);
        dialog.show();
        QTest::qWait(10);
        populateDatabase(backend);
        QSignalSpy removed(&dialog, &FaceDatabaseDialog::deletePersonRequested);
        auto *button = deleteButton(dialog, QString::fromUtf8("删除人员…"));
        QVERIFY(button && button->isEnabled());
        bool confirmationShown = false;
        QTimer::singleShot(0, &dialog, [&] {
            auto *box = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
            confirmationShown = box != nullptr;
            if (!box) return;
            backend.faceDatabaseChanged({{{"P2", "Two", "out"}, true, 1}});
            box->button(QMessageBox::Yes)->click();
        });
        button->click();
        QVERIFY(confirmationShown);
        QCOMPARE(removed.count(), 1);
        QCOMPARE(removed.first()[0].toString(), QString("P1"));
        backend.shutdown();
    }

    void sampleDeletionSurvivesPreviewRefreshDuringConfirmation() {
        MockFaceBackend backend;
        FaceDatabaseDialog dialog(&backend);
        dialog.show();
        QTest::qWait(10);
        populateDatabase(backend);
        auto *samples = dialog.findChild<QListWidget *>("databaseSamples");
        samples->setCurrentRow(0);
        QSignalSpy removed(&dialog, &FaceDatabaseDialog::deleteSampleRequested);
        auto *button = deleteButton(dialog, QString::fromUtf8("删除选中样本…"));
        QVERIFY(button && button->isEnabled());
        bool confirmationShown = false;
        QTimer::singleShot(0, &dialog, [&] {
            auto *box = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
            confirmationShown = box != nullptr;
            if (!box) return;
            FaceDatabaseSample replacement;
            replacement.sampleId = "S2";
            backend.personSamplesChanged("P1", {replacement});
            backend.faceDatabaseChanged({{{"P2", "Two", "out"}, true, 1}});
            box->button(QMessageBox::Yes)->click();
        });
        button->click();
        QVERIFY(confirmationShown);
        QCOMPARE(removed.count(), 1);
        QCOMPARE(removed.first()[0].toString(), QString("P1"));
        QCOMPARE(removed.first()[1].toString(), QString("S1"));
        backend.shutdown();
    }

    void cancellingSampleDeletionAfterPreviewRefreshEmitsNothing() {
        MockFaceBackend backend;
        FaceDatabaseDialog dialog(&backend);
        dialog.show();
        QTest::qWait(10);
        populateDatabase(backend);
        dialog.findChild<QListWidget *>("databaseSamples")->setCurrentRow(0);
        QSignalSpy removed(&dialog, &FaceDatabaseDialog::deleteSampleRequested);
        auto *button = deleteButton(dialog, QString::fromUtf8("删除选中样本…"));
        QVERIFY(button && button->isEnabled());
        bool confirmationShown = false;
        QTimer::singleShot(0, &dialog, [&] {
            auto *box = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
            confirmationShown = box != nullptr;
            if (!box) return;
            backend.personSamplesChanged("P1", {});
            box->button(QMessageBox::No)->click();
        });
        button->click();
        QVERIFY(confirmationShown);
        QCOMPARE(removed.count(), 0);
        backend.shutdown();
    }

    void managementSearchEditingAndPreviewRespectSelection() {
        MockFaceBackend backend;
        FaceDatabaseDialog dialog(&backend);
        dialog.show();
        QTest::qWait(10); // Drain the initial queued refresh before supplying the fixture.
        const FaceDatabasePeople people{{{"P001", QString::fromUtf8("张三"), "in"}, true, 10},
                                        {{"P002", QString::fromUtf8("李四"), "out"}, false, 20}};
        backend.faceDatabaseChanged(people);
        auto *table = dialog.findChild<QTableWidget *>("databasePeople");
        auto *search = dialog.findChild<QLineEdit *>("databaseSearch");
        auto *name = dialog.findChild<QLineEdit *>("databasePersonName");
        auto *samples = dialog.findChild<QListWidget *>("databaseSamples");
        QCOMPARE(table->rowCount(), 2);
        search->setText(QString::fromUtf8("李四"));
        QVERIFY(table->isRowHidden(0)); QVERIFY(!table->isRowHidden(1));
        table->selectRow(1);
        QCOMPARE(name->text(), QString::fromUtf8("李四"));
        QVERIFY(!dialog.findChild<QCheckBox *>("databasePersonEnabled")->isChecked());
        FaceDatabaseSample fixture;
        fixture.sampleId = "S1"; fixture.sessionId = "CAM1";
        fixture.image = QImage(112,112,QImage::Format_RGB32);
        fixture.image.fill(QColor(176,192,207));
        QPainter painter(&fixture.image);
        painter.setPen(Qt::NoPen); painter.setBrush(QColor(91,112,132));
        painter.drawEllipse(QRect(36,20,40,40)); painter.drawRoundedRect(QRect(24,64,64,48),20,20);
        painter.end();
        backend.personSamplesChanged("P001", {fixture});
        QCOMPARE(samples->count(), 0); // A late reply for another person cannot change the preview.
        backend.personSamplesChanged("P002", {fixture});
        QCOMPARE(samples->count(), 1);
        name->setText(QString::fromUtf8("李四新姓名"));
        QSignalSpy saved(&dialog, &FaceDatabaseDialog::updatePersonRequested);
        dialog.findChild<QPushButton *>("databaseSavePerson")->click();
        QCOMPARE(saved.count(), 1);
        const auto profile = saved.first()[0].value<PersonProfile>();
        QCOMPARE(profile.personId, QString("P002"));
        QCOMPARE(profile.displayName, QString::fromUtf8("李四新姓名"));
        QVERIFY(!saved.first()[1].toBool());
        backend.faceDatabaseChanged(people);
        search->clear(); table->selectRow(0);
        QTest::qWait(10);
        backend.personSamplesChanged("P001", {fixture,fixture,fixture});
        const auto screenshot = qgetenv("FACEAPP_TEST_SCREENSHOT_DIR");
        if (!screenshot.isEmpty()) {
            QDir().mkpath(QString::fromUtf8(screenshot));
            dialog.resize(1240,800);
            QTest::qWait(20);
            QCOMPARE(samples->count(), 3);
            QVERIFY(dialog.grab().save(QString::fromUtf8(screenshot)+"/face-database.png"));
        }
        dialog.close(); backend.shutdown();
    }
};
QTEST_MAIN(FeatureUiTests)
#include "feature_ui_tests.moc"
