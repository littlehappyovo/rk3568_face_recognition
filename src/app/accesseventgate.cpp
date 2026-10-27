#include "accesseventgate.h"
#include <QUuid>

QVector<QJsonObject> AccessEventGate::observe(const QString &stationId, const QString &streamId,
    int directionVersion, const QVector<FaceObservation> &faces,
    const QSet<int> &retainedTrackIds, const QDateTime &observedUtc, bool liveInput)
{
    QVector<QJsonObject> events;
    if (!liveInput || directionVersion < 1 || stationId.isEmpty() || QUuid(streamId).isNull()
        || !observedUtc.isValid()) return events;
    auto &source = m_sources[stationId];
    if (source.streamId != streamId || source.version != directionVersion) {
        source = Source{streamId, directionVersion, {}};
    }
    for (auto it = source.trackEvents.begin(); it != source.trackEvents.end();) {
        if (!retainedTrackIds.contains(it.key())) it = source.trackEvents.erase(it);
        else ++it;
    }
    for (const auto &face : faces) {
        if (face.trackId < 0 || !retainedTrackIds.contains(face.trackId)
            || face.identityState == IdentityState::DetectionOnly) continue;
        const bool allowed = face.identityState == IdentityState::Known && !face.personId.isEmpty();
        auto previous = source.trackEvents.find(face.trackId);
        if (previous != source.trackEvents.end()
            && (previous->allowed || (!allowed && previous->denied))) continue;
        events.push_back({{"event_id", QUuid::createUuid().toString(QUuid::WithoutBraces)},
            {"station_id", stationId}, {"direction_version", directionVersion},
            {"stream_id", streamId}, {"track_id", face.trackId},
            {"observed_at", observedUtc.toUTC().toString(Qt::ISODateWithMs)},
            {"person_id", allowed ? QJsonValue(face.personId) : QJsonValue(QJsonValue::Null)},
            {"display_name", allowed ? face.displayName : QString()},
            {"decision", allowed ? QStringLiteral("allowed") : QStringLiteral("denied")}});
        auto &trackEvents = source.trackEvents[face.trackId];
        if (allowed) trackEvents.allowed = true;
        else trackEvents.denied = true;
    }
    return events;
}

void AccessEventGate::resetSource(const QString &stationId) { m_sources.remove(stationId); }
