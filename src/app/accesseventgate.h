#pragma once

#include "backendtypes.h"
#include <QDateTime>
#include <QHash>
#include <QJsonObject>
#include <QSet>

class AccessEventGate
{
public:
    QVector<QJsonObject> observe(const QString &stationId, const QString &streamId,
        int directionVersion, const QVector<FaceObservation> &faces,
        const QSet<int> &retainedTrackIds, const QDateTime &observedUtc, bool liveInput = true);
    void resetSource(const QString &stationId);
private:
    struct TrackEvents { bool denied = false; bool allowed = false; };
    struct Source { QString streamId; int version = 0; QHash<int, TrackEvents> trackEvents; };
    QHash<QString, Source> m_sources;
};
