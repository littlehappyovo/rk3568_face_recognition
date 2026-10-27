#include "accessclient.h"
#include "accessruntime.h"
#include "adminlogindialog.h"
#include "mainwindow.h"
#include "mockfacebackend.h"
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QProcess>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalSpy>
#include <QTcpServer>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QWindow>
#include <memory>

class AdminUiTests : public QObject {
    Q_OBJECT
    std::unique_ptr<QTemporaryDir> m_data;
    std::unique_ptr<AccessRuntime> m_runtime;
    std::unique_ptr<AccessClient> m_client;
    std::unique_ptr<MockFaceBackend> m_backend;
    std::unique_ptr<MainWindow> m_window;
    QUrl m_url;
private slots:
    void initTestCase() { registerFaceBackendMetaTypes(); }
    void init() {
        m_data = std::make_unique<QTemporaryDir>(); qputenv("FACEAPP_ACCESS_PORT", "0");
        m_runtime = std::make_unique<AccessRuntime>(QString::fromUtf8(FACE_TEST_PROJECT_ROOT));
        m_client = std::make_unique<AccessClient>();
        m_backend = std::make_unique<MockFaceBackend>();
        m_window = std::make_unique<MainWindow>(m_backend.get(), false);
        m_window->setAccessClient(m_client.get()); m_window->show();
        connect(m_runtime.get(), &AccessRuntime::ready, m_client.get(), [this](const QUrl &url) {
            m_url = url; m_client->setEndpoint(url, m_runtime->localSecret());
        });
        QSignalSpy ready(m_runtime.get(), &AccessRuntime::ready);
        m_runtime->start(m_data->path(), QString::fromUtf8(FACE_TEST_ACCESS_PYTHON));
        QTRY_COMPARE_WITH_TIMEOUT(ready.count(), 1, 5000);
    }
    void cleanup() {
        m_window.reset(); m_client.reset(); m_backend->shutdown(); m_backend.reset();
        m_runtime->stop(); m_runtime.reset(); m_data.reset(); qunsetenv("FACEAPP_ACCESS_PORT");
    }
    void firstSetupRequiresMatchingPasswordAndServerApproval() {
        m_client->setEndpoint(m_url, "wrong-startup-secret");
        m_window->findChild<QPushButton *>("adminButton")->click();
        auto *dialog = m_window->findChild<AdminLoginDialog *>(); QVERIFY(dialog);
        auto *password = dialog->findChild<QLineEdit *>("adminPassword");
        auto *confirm = dialog->findChild<QLineEdit *>("adminPasswordConfirm");
        auto *submit = dialog->findChild<QPushButton *>("adminSubmit");
        QTRY_VERIFY(confirm->isVisible());
        password->setText("Test-Password-2026"); confirm->setText("mismatch-password"); submit->click();
        QVERIFY(m_window->findChild<QScrollArea *>("managementControls")->isHidden());
        confirm->setText("Test-Password-2026"); submit->click();
        QTRY_VERIFY(submit->isEnabled());
        QVERIFY(dialog->isVisible());
        QVERIFY(m_window->findChild<QScrollArea *>("managementControls")->isHidden());
        m_client->setEndpoint(m_url, m_runtime->localSecret()); submit->click();
        QTRY_VERIFY_WITH_TIMEOUT(m_window->findChild<QScrollArea *>("managementControls")->isVisible(), 4000);
    }
    void failedLoginKeepsAdministrationLocked() {
        QSignalSpy authenticated(m_client.get(), &AccessClient::authenticated);
        m_client->setupPassword("Test-Password-2026"); QTRY_COMPARE(authenticated.count(), 1);
        m_client->logout();
        m_window->findChild<QPushButton *>("adminButton")->click();
        auto *dialog = m_window->findChild<AdminLoginDialog *>(); QVERIFY(dialog);
        auto *submit = dialog->findChild<QPushButton *>("adminSubmit");
        QTRY_VERIFY(submit->isEnabled());
        dialog->findChild<QLineEdit *>("adminPassword")->setText("incorrect-password"); submit->click();
        QTRY_VERIFY(submit->isEnabled());
        QVERIFY(dialog->isVisible());
        QVERIFY(m_window->findChild<QScrollArea *>("managementControls")->isHidden());
    }
    void physicalKeyboardLogsInWithoutOnscreenKeys() {
        QSignalSpy authenticated(m_client.get(), &AccessClient::authenticated);
        m_client->setupPassword("Test-Password-2026"); QTRY_COMPARE(authenticated.count(), 1);
        m_client->logout();
        m_window->findChild<QPushButton *>("adminButton")->click();
        auto *dialog = m_window->findChild<AdminLoginDialog *>(); QVERIFY(dialog);
        QTRY_VERIFY(dialog->findChild<QPushButton *>("adminSubmit")->isEnabled());
        QVERIFY(!dialog->findChild<QPushButton *>("key_q"));
        auto *password = dialog->findChild<QLineEdit *>("adminPassword");
        QTRY_COMPARE(QApplication::focusWidget(), password);
        QTest::keyClicks(QApplication::focusWidget(), "Test-Password-2026");
        QTest::keyClick(m_window->windowHandle(), Qt::Key_Return);
        QTRY_VERIFY_WITH_TIMEOUT(m_window->findChild<QScrollArea *>("managementControls")->isVisible(), 4000);
    }
    void revokedSessionCannotRunLocalTraining() {
        QSignalSpy authenticated(m_client.get(), &AccessClient::authenticated);
        m_client->setupPassword("Test-Password-2026"); QTRY_COMPARE(authenticated.count(), 1);
        m_window->setAdminSession(m_client->sessionId(), m_client->expiresAt());
        QProcess revoke;
        revoke.start(QString::fromUtf8(FACE_TEST_ACCESS_PYTHON), {"-I", "-c",
            "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute('DELETE FROM auth_sessions'); c.commit()",
            m_data->filePath("access.sqlite3")});
        QVERIFY(revoke.waitForFinished(3000)); QCOMPARE(revoke.exitCode(), 0);
        QSignalSpy training(m_backend.get(), &FaceBackendService::trainingProgress);
        for (auto *button : m_window->findChildren<QPushButton *>())
            if (button->text() == QString::fromUtf8("训练 / 重建模型")) button->click();
        QTRY_VERIFY(m_window->findChild<QScrollArea *>("managementControls")->isHidden());
        QCOMPARE(training.count(), 0);
    }
    void backendUnavailableKeepsCameraAndUiResponsive() {
        QTcpServer unused; QVERIFY(unused.listen(QHostAddress::LocalHost,0));
        const auto url = QUrl(QString("http://127.0.0.1:%1").arg(unused.serverPort())); unused.close();
        m_client->setEndpoint(url, "test-startup-secret");
        int ticks = 0; QTimer heartbeat; heartbeat.setInterval(1);
        connect(&heartbeat, &QTimer::timeout, this, [&] { ++ticks; }); heartbeat.start();
        m_window->findChild<QPushButton *>("adminButton")->click();
        auto *dialog = m_window->findChild<AdminLoginDialog *>(); QVERIFY(dialog);
        QTRY_VERIFY(!dialog->findChild<QLabel *>("adminError")->text().isEmpty());
        QTRY_VERIFY(ticks > 5); heartbeat.stop();
        QVERIFY(m_window->findChild<QScrollArea *>("managementControls")->isHidden());
    }

};
QTEST_MAIN(AdminUiTests)
#include "admin_ui_tests.moc"
