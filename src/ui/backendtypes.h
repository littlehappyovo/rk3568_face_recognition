#pragma once

#include <QColor>
#include <QImage>
#include <QMetaType>
#include <QRectF>
#include <QString>
#include <QVector>

enum class ServiceState {
    Idle,
    Streaming,
    Capturing,
    Importing,
    Training,
    Error
};

enum class IdentityState {
    DetectionOnly,
    Known,
    Unknown
};

struct CameraDescriptor {
    QString id;
    QString displayName;
};

struct PersonProfile {
    QString personId;
    QString displayName;
    QString group;
};

struct FaceDatabasePerson {
    PersonProfile profile;
    bool enabled = true;
    int sampleCount = 0;
};

struct FaceDatabaseSample {
    QString sampleId;
    QString sessionId;
    QString details;
    QImage image;
};

using FaceDatabasePeople = QVector<FaceDatabasePerson>;
using FaceDatabaseSamples = QVector<FaceDatabaseSample>;

struct FaceObservation {
    // Coordinates are normalized to [0, 1] relative to the processed image.
    QRectF normalizedBox;
    int trackId = -1;
    IdentityState identityState = IdentityState::DetectionOnly;
    QString personId;
    QString displayName;
    QString group;
    QString scoreText;
    QColor color;
    bool selectedForCapture = false;
};

struct ReferenceFaceCard {
    QImage image;
    QString personId;
    QString displayName;
    QString group;
};

struct SystemMetrics {
    // Values are percentages unless the field name says Celsius. Negative means unavailable.
    double cpuUsagePercent = -1.0;
    double npuUsagePercent = -1.0;
    double temperatureCelsius = -1.0;
    double memoryUsagePercent = -1.0;
};

struct FrameResult {
    QImage originalImage;
    // The UI adds boxes/labels to this image. The backend must not draw them twice.
    QImage processedImage;
    QVector<FaceObservation> faces;
    QVector<ReferenceFaceCard> referenceCards;
    qint64 referenceRevision = 0;
    double fps = 0.0;
    qint64 sequence = 0;
    SystemMetrics systemMetrics;
};

using CameraList = QVector<CameraDescriptor>;

Q_DECLARE_METATYPE(ServiceState)
Q_DECLARE_METATYPE(IdentityState)
Q_DECLARE_METATYPE(CameraDescriptor)
Q_DECLARE_METATYPE(CameraList)
Q_DECLARE_METATYPE(PersonProfile)
Q_DECLARE_METATYPE(FaceDatabasePeople)
Q_DECLARE_METATYPE(FaceDatabaseSamples)
Q_DECLARE_METATYPE(FaceObservation)
Q_DECLARE_METATYPE(ReferenceFaceCard)
Q_DECLARE_METATYPE(SystemMetrics)
Q_DECLARE_METATYPE(FrameResult)

inline void registerFaceBackendMetaTypes()
{
    qRegisterMetaType<ServiceState>("ServiceState");
    qRegisterMetaType<IdentityState>("IdentityState");
    qRegisterMetaType<CameraDescriptor>("CameraDescriptor");
    qRegisterMetaType<CameraList>("CameraList");
    qRegisterMetaType<PersonProfile>("PersonProfile");
    qRegisterMetaType<FaceDatabasePeople>("FaceDatabasePeople");
    qRegisterMetaType<FaceDatabaseSamples>("FaceDatabaseSamples");
    qRegisterMetaType<FaceObservation>("FaceObservation");
    qRegisterMetaType<ReferenceFaceCard>("ReferenceFaceCard");
    qRegisterMetaType<SystemMetrics>("SystemMetrics");
    qRegisterMetaType<FrameResult>("FrameResult");
}
