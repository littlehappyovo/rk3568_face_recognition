#pragma once

#include "backendtypes.h"

#include <QMainWindow>

class FaceBackendService;
class FrameView;
class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QSlider;

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(FaceBackendService *backend, QWidget *parent = nullptr);
    MainWindow(FaceBackendService *backend, bool autoStart, QWidget *parent = nullptr);

signals:
    void configureBackendRequested(const QString &backendId);
    void refreshCamerasRequested();
    void openCameraRequested(const QString &cameraId);
    void closeInputRequested();
    void importImagesRequested(const QStringList &paths,
                               const PersonProfile &person,
                               bool saveToDataset);
    void beginCaptureRequested(const PersonProfile &person, int targetSamples);
    void selectCaptureTrackRequested(qint64 trackId);
    void endCaptureRequested();
    void beginRawFrameCaptureRequested();
    void trainModelRequested();
    void recognitionRequested(bool enabled);
    void autoStartRequested();
    void voiceVolumeRequested(int volume, bool muted);

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

private slots:
    void chooseImages();
    void startSelectedCamera();
    void beginCapture();
    void onCamerasChanged(const CameraList &cameras);
    void onFrameReady(const FrameResult &frame);
    void onStateChanged(ServiceState state, const QString &message);
    void onCaptureProgress(int acceptedSamples, int targetSamples);
    void onRawCaptureProgress(int capturedFrames, int targetFrames);
    void onTrainingProgress(int percent, const QString &message);
    void onError(const QString &title, const QString &details);

private:
    void buildUi();
    void connectBackend(FaceBackendService *backend);
    void updateControls();
    QString stateName(ServiceState state) const;

    FaceBackendService *m_backend = nullptr;
    FrameView *m_originalView = nullptr;
    FrameView *m_recognizedView = nullptr;
    QComboBox *m_backendCombo = nullptr;
    QComboBox *m_cameraCombo = nullptr;
    QPushButton *m_refreshButton = nullptr;
    QPushButton *m_startButton = nullptr;
    QPushButton *m_stopButton = nullptr;
    QPushButton *m_importButton = nullptr;
    QLineEdit *m_personIdEdit = nullptr;
    QLineEdit *m_personNameEdit = nullptr;
    QComboBox *m_groupCombo = nullptr;
    QSpinBox *m_sampleCountSpin = nullptr;
    QPushButton *m_captureButton = nullptr;
    QPushButton *m_endCaptureButton = nullptr;
    QPushButton *m_rawCaptureButton = nullptr;
    QPushButton *m_trainButton = nullptr;
    QPushButton *m_databaseButton = nullptr;
    QCheckBox *m_recognitionCheck = nullptr;
    QSlider *m_voiceVolume = nullptr;
    QCheckBox *m_voiceMute = nullptr;
    QLabel *m_voiceVolumeLabel = nullptr;
    QProgressBar *m_captureProgress = nullptr;
    QProgressBar *m_rawCaptureProgress = nullptr;
    QProgressBar *m_trainingProgress = nullptr;
    QLabel *m_statusLabel = nullptr;
    QLabel *m_fpsLabel = nullptr;
    QLabel *m_targetCountLabel = nullptr;
    QLabel *m_modelLabel = nullptr;

    ServiceState m_state = ServiceState::Idle;
    bool m_streaming = false;
    bool m_capturing = false;
    bool m_rawCapturing = false;
    bool m_training = false;
    qint64 m_referenceRevision = -1;
};
