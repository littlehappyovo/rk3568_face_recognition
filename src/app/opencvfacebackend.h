#pragma once

#include "facebackendservice.h"
#include "recognitionmemory.h"

#include <facecore/facecore.hpp>

#include <QElapsedTimer>

#include <atomic>
#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

class QTimer;
class VoiceAnnouncer;

// Qt 层和 facecore 核心库之间的业务适配器。
//
// main.cpp 会把这个 QObject 移到一个工作 QThread。它负责摄像头、检测、跟踪、
// 采集和训练，但不直接操作 QWidget；界面只通过信号槽拿到 FrameResult。
class OpenCvFaceBackend final : public FaceBackendService
{
    Q_OBJECT

public:
    explicit OpenCvFaceBackend(QString applicationDirectory, QObject *parent = nullptr);
    ~OpenCvFaceBackend() override;
    void requestShutdown() noexcept override;

public slots:
    void setVoiceVolume(int volume, bool muted) override;
    void configureBackend(const QString &backendId) override;
    void refreshCameras() override;
    void startAutomatically() override;
    void refreshFaceDatabase() override;
    void loadPersonSamples(const QString &personId) override;
    void updateDatabasePerson(const PersonProfile &person, bool enabled) override;
    void deleteDatabasePerson(const QString &personId) override;
    void deleteDatabaseSample(const QString &personId, const QString &sampleId) override;
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
    void processNextFrame();

private:
    friend class BackendDatabaseTests;
    class AsyncFrameReader;
    class AsyncDetector;

    using RecognitionMemory = TrackRecognitionMemory;

    void initializeCore();
    void requireDatabaseWritable();
    void finishDatabaseChange(const QString &message);
    void completeGalleryRebuild(const std::string &backendId,
                                const std::filesystem::path &modelFile);
    void loadRegistry();
    void loadNewestModels();
    void clearReferenceCards() noexcept;
    QImage referenceImageForPerson(const std::string &personId);
    void registerPersonOrThrow(const PersonProfile &person);
    FrameResult processFrame(const facecore::Frame &frame, bool allowCapture,
                            const std::vector<facecore::FaceDetection> *imageDetections = nullptr);
    void updateSystemMetrics();
    std::vector<facecore::TrackedFace> updateTracks(const cv::Mat &image,
                                                    std::uint64_t sequence);
    std::vector<facecore::FaceDetection> predictDetections(
        const cv::Mat &previousGray, const cv::Mat &currentGray) const;
    void resetTrackingState() noexcept;
    FaceObservation makeObservation(const facecore::TrackedFace &track,
                                    const facecore::QualityReport &quality,
                                    const std::optional<facecore::RecognitionResult> &recognition) const;
    bool recognitionDue(const RecognitionMemory &memory, std::int64_t timestampUs) const noexcept;
    std::optional<facecore::RecognitionResult> cachedRecognition(
        std::uint64_t trackId) const;
    std::optional<facecore::RecognitionResult> recognizeTrack(
        const facecore::TrackedFace &track,
        const facecore::FaceSample &sample,
        std::uint64_t sequence,
        std::int64_t timestampUs,
        bool temporalVoting = true);
    void captureIfEligible(const facecore::Frame &frame,
                           const std::vector<facecore::TrackedFace> &tracks);
    void captureRawFrameIfRequested(const facecore::Frame &frame);
    void saveSample(const facecore::Frame &frame,
                    const facecore::TrackedFace &track,
                    const facecore::FaceSample &sample,
                    const facecore::QualityReport &quality,
                    const PersonProfile &person,
                    const std::string &sessionId,
                    const std::string &source);
    bool backendReady() const;
    void setState(ServiceState state, const QString &message);
    void reportException(const QString &title, const std::exception &exception);
    void throwIfShutdownRequested() const;
    static std::filesystem::path nativePath(const QString &path);
    static QString utf8(const std::string &text);
    static QImage toQImage(const cv::Mat &image);
    static facecore::PersonRecord toPersonRecord(const PersonProfile &person);
    static QString groupName(facecore::PersonGroup group);
    static std::string makeSessionId(const char *prefix);
    static QString qualityMessage(const facecore::QualityReport &quality);
    static std::vector<std::filesystem::path> modelFilesNewestFirst(
        const std::filesystem::path &directory,
        const std::string &prefix,
        const std::string &extension);

    QString m_applicationDirectory;
    std::unique_ptr<VoiceAnnouncer> m_voice;
    std::filesystem::path m_runtimeDirectory;
    QString m_backendId = QStringLiteral("mobilefacenet_rknn");
    // embedding 后端的实际状态说明（NPU / CPU / 模型待投放），用于界面提示。
    QString m_backendSummary;
    QString m_mobileFaceNetSummary;
    QTimer *m_frameTimer = nullptr;
    QElapsedTimer m_systemMetricsClock;
    QElapsedTimer m_frameCadenceClock;
    qint64 m_lastSystemMetricsMs = -1;
    std::uint64_t m_previousCpuTotal = 0;
    std::uint64_t m_previousCpuIdle = 0;
    SystemMetrics m_systemMetrics;
    // m_cameraReader 自己拥有采集线程，只保留一个最新帧；m_asyncDetector 自己
    // 拥有检测线程，只允许一个待处理任务和一个未消费结果。
    std::unique_ptr<AsyncFrameReader> m_cameraReader;
    std::unique_ptr<AsyncDetector> m_asyncDetector;
    std::unique_ptr<facecore::FaceTracker> m_tracker;
    std::unique_ptr<facecore::FacePreprocessor> m_preprocessor;
    std::unique_ptr<facecore::QualityEvaluator> m_quality;
    std::unique_ptr<facecore::DuplicateFilter> m_duplicateFilter;
    std::unique_ptr<facecore::DatasetStore> m_dataset;
    std::unique_ptr<facecore::LbphFaceRecognizer> m_lbph;
    std::shared_ptr<facecore::IEmbeddingFaceRecognizer> m_sface;
    std::shared_ptr<facecore::IEmbeddingFaceRecognizer> m_mobileFaceNet;
    facecore::SystemConfig m_config;
    std::unordered_map<std::string, facecore::PersonRecord> m_people;
    std::unordered_map<std::string, facecore::SampleRecord> m_referenceSamples;
    std::unordered_map<std::string, QImage> m_referenceImageCache;
    std::unordered_set<std::string> m_referenceImageMisses;
    QVector<ReferenceFaceCard> m_referenceCards;
    std::vector<std::string> m_referenceCardPersonIds;
    std::uint64_t m_referenceRevision = 0;
    std::unordered_map<std::uint64_t, RecognitionMemory> m_recognitionMemory;
    std::unordered_set<std::uint64_t> m_activeTrackIds;
    std::vector<facecore::TrackedFace> m_lastTracks;
    cv::Mat m_previousTrackingGray;
    std::uint64_t m_lastDetectionSequence = 0;
    std::uint64_t m_lastDetectionRequestSequence = 0;
    int m_consecutiveDetectionMisses = 0;

    PersonProfile m_capturePerson;
    std::string m_captureSessionId;
    std::optional<std::uint64_t> m_selectedTrackId;
    std::int64_t m_lastCaptureTimestampUs = 0;
    std::uint64_t m_sequence = 0;
    int m_captureAccepted = 0;
    int m_captureTarget = 0;
    std::string m_rawCaptureSessionId;
    std::int64_t m_lastRawCaptureTimestampUs = 0;
    int m_rawCaptureAccepted = 0;
    int m_rawCaptureTarget = 0;
    double m_smoothedFps = 0.0;
    bool m_initialized = false;
    bool m_streaming = false;
    bool m_captureActive = false;
    bool m_rawCaptureActive = false;
    bool m_training = false;
    bool m_recognitionEnabled = false;
    std::atomic_bool m_shutdownRequested{false};
};
