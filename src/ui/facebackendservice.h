#pragma once

#include "backendtypes.h"

#include <QObject>
#include <QStringList>
#include <QJsonObject>

// Thread boundary between Qt Widgets and the camera/detection/recognition core.
// Implementations live on a worker QThread. The UI only talks to these slots
// through queued signals and never accesses OpenCV objects directly.
class FaceBackendService : public QObject
{
    Q_OBJECT

public:
    explicit FaceBackendService(QObject *parent = nullptr)
        : QObject(parent)
    {
    }

    ~FaceBackendService() override = default;

    // The application calls this directly before waiting for the worker thread.
    // Implementations must only touch atomics here; all QObject cleanup stays in shutdown().
    virtual void requestShutdown() noexcept = 0;

signals:
    void camerasChanged(const CameraList &cameras);
    void frameReady(const FrameResult &frame);
    void stateChanged(ServiceState state, const QString &message);
    void captureProgress(int acceptedSamples, int targetSamples);
    void rawCaptureProgress(int capturedFrames, int targetFrames);
    void trainingProgress(int percent, const QString &message);
    void modelChanged(const QString &modelVersion, const QString &summary);
    void recognitionEnabledChanged(bool enabled);
    void voiceVolumeChanged(int volume, bool muted);
    void errorOccurred(const QString &title, const QString &details);
    void faceDatabaseChanged(const FaceDatabasePeople &people);
    void personSamplesChanged(const QString &personId, const FaceDatabaseSamples &samples);
    void databaseOperationFinished(const QString &message);
    void stableAccessEvent(const QJsonObject &event);
    void stationConfigured(const QString &stationId, const QString &direction, int version, bool enabled);
    void accessCommandFinished(const QString &requestId, const QJsonObject &result, const QJsonObject &error);

public slots:
    virtual void executeAccessCommand(const QString &requestId, const QString &, const QJsonObject &) {
        emit accessCommandFinished(requestId, {}, {{"code", "unsupported"}, {"message", tr("当前后端不支持远程管理")}});
    }
    virtual void setVoiceVolume(int volume, bool muted) { emit voiceVolumeChanged(volume, muted); }
    virtual void configureBackend(const QString &backendId) = 0;
    virtual void refreshCameras() = 0;
    virtual void startAutomatically() = 0;
    virtual void refreshFaceDatabase() { emit faceDatabaseChanged({}); }
    virtual void loadPersonSamples(const QString &personId) {
        emit personSamplesChanged(personId, {});
    }
    virtual void updateDatabasePerson(const PersonProfile &, bool) {
        emit errorOccurred(tr("演示模式"), tr("演示模式不修改真实人脸库。"));
    }
    virtual void deleteDatabasePerson(const QString &) {
        emit errorOccurred(tr("演示模式"), tr("演示模式不修改真实人脸库。"));
    }
    virtual void deleteDatabaseSample(const QString &, const QString &) {
        emit errorOccurred(tr("演示模式"), tr("演示模式不修改真实人脸库。"));
    }
    virtual void openCamera(const QString &cameraId) = 0;
    virtual void closeInput() = 0;
    virtual void importImages(const QStringList &paths,
                              const PersonProfile &person,
                              bool saveToDataset) = 0;
    virtual void beginCapture(const PersonProfile &person, int targetSamples) = 0;
    virtual void selectCaptureTrack(qint64 trackId) = 0;
    virtual void endCapture() = 0;
    virtual void beginRawFrameCapture() = 0;
    virtual void trainModel() = 0;
    virtual void setRecognitionEnabled(bool enabled) = 0;
    virtual void shutdown() = 0;
};
