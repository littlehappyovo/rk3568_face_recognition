#pragma once
#include "backendtypes.h"
#include <QObject>
#include <QProcess>
#include <QQueue>
#include <QSet>
#include <QTimer>
#include <QHash>

struct VoiceSettings {
    bool enabled = true;
    QString audioDirectory;
    QString device = QStringLiteral("plughw:CARD=rockchiprk809,DEV=0");
    QString playerProgram = QStringLiteral("aplay");
    QString mixerProgram = QStringLiteral("amixer");
    QString pulseProgram = QStringLiteral("paplay");
    QString pulseServer;
    QString pulseSink = QStringLiteral("alsa_output.platform-rk809-sound.stereo-fallback");
    QString preferencesFile;
    bool preferSystemAudio = false;
    bool hardwareVolume = false;
    int volume = 90;
    bool muted = false;
    int morning = 5, noon = 11, afternoon = 14, evening = 18;
    static VoiceSettings load(const QString &baseDirectory);
};

class VoiceAnnouncer : public QObject {
    Q_OBJECT
public:
    explicit VoiceAnnouncer(VoiceSettings settings, QObject *parent = nullptr);
    ~VoiceAnnouncer() override;
    static QString phraseForHour(int hour, const VoiceSettings &settings = {});
    void onFrame(const QVector<FaceObservation> &faces, int hour);
    void onFrame(const QVector<FaceObservation> &faces, int hour, const QSet<int> &aliveTracks);
    void reset();
    void setVolume(int volume, bool muted);
    int volume() const { return m_settings.volume; }
    bool muted() const { return m_settings.muted; }
signals:
    void playbackRequested(int trackId, const QString &phrase, const QString &file);
    void playbackFailed(const QString &message);
    void volumeChanged(int volume, bool muted);
private:
    struct Request { int trackId; QString phrase; };
    void playNext();
    void startCurrent(bool systemAudio);
    void finishPlayback(bool failed, const QString &message);
    void applyVolume();
    VoiceSettings m_settings;
    QSet<int> m_announced;
    QSet<int> m_active;
    QHash<int,QString> m_identities;
    QQueue<Request> m_pending;
    QProcess m_player;
    QProcess m_mixer;
    QTimer m_timeout;
    QTimer m_mixerTimeout;
    QString m_currentFile;
    QString m_currentIdentity;
    quint64 m_playbackGeneration = 0;
    bool m_usingPulse = false;
    bool m_volumeDirty = true;
    int m_playingTrack = -1;
};
