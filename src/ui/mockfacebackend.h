#pragma once

#include "facebackendservice.h"

#include <QElapsedTimer>

#include <atomic>

class QTimer;

// Demonstration backend. Replace this class with an adapter around the real
// OpenCV core; its output is synthetic and must not be used as test evidence.
class MockFaceBackend final : public FaceBackendService
{
    Q_OBJECT

public:
    explicit MockFaceBackend(QObject *parent = nullptr);
    void requestShutdown() noexcept override;

public slots:
    void configureBackend(const QString &backendId) override;
    void refreshCameras() override;
    void startAutomatically() override;
    void openCamera(const QString &cameraId) override;
    void closeInput() override;
    void importImages(const QStringList &paths,
                      const PersonProfile &person,
                      bool saveToDataset) override;
    void beginCapture(const PersonProfile &person, int targetSamples) override;
    void selectCaptureTrack(qint64 trackId) override;
    void endCapture() override;
    void beginRawFrameCapture() override;
    void trainModel() override;
    void setRecognitionEnabled(bool enabled) override;
    void shutdown() override;

private slots:
    void generateFrame();
    void advanceTraining();

private:
    FrameResult makeDemoFrame();
    void setState(ServiceState state, const QString &message);

    QTimer *m_frameTimer = nullptr;
    QTimer *m_trainingTimer = nullptr;
    QString m_backendId = QStringLiteral("opencv_lbph");
    QString m_cameraId;
    PersonProfile m_capturePerson;
    qint64 m_sequence = 0;
    int m_captureAccepted = 0;
    int m_captureTarget = 0;
    int m_rawCaptureAccepted = 0;
    int m_rawCaptureTarget = 0;
    int m_trainingPercent = 0;
    bool m_running = false;
    bool m_captureActive = false;
    bool m_rawCaptureActive = false;
    bool m_training = false;
    bool m_recognitionEnabled = false;
    std::atomic_bool m_shutdownRequested{false};
};
