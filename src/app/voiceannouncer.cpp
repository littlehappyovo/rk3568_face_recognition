#include "voiceannouncer.h"
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSettings>
#include <QStandardPaths>
#ifdef Q_OS_UNIX
#include <unistd.h>
#endif
#include <algorithm>
#include <utility>

VoiceSettings VoiceSettings::load(const QString &baseDirectory)
{
    VoiceSettings settings;
    settings.hardwareVolume = true;
    settings.preferSystemAudio = true;
#ifdef Q_OS_UNIX
    settings.pulseServer = QStringLiteral("/run/user/%1/pulse/native").arg(getuid());
#endif
    settings.audioDirectory = QDir(baseDirectory).filePath("audio");
    const QString configured = qEnvironmentVariable("FACEAPP_VOICE_CONFIG");
    QFile file(configured.isEmpty() ? QDir(baseDirectory).filePath("config/voice.json") : configured);
    if (file.exists() || !configured.isEmpty()) {
        QJsonParseError error;
        if (!file.open(QIODevice::ReadOnly)) {
            qWarning() << "[voice] cannot read configuration" << file.fileName();
            settings.enabled = false;
            return settings;
        }
        const auto doc = QJsonDocument::fromJson(file.readAll(), &error);
        if (error.error != QJsonParseError::NoError || !doc.isObject()) {
            qWarning() << "[voice] invalid configuration" << file.fileName();
            settings.enabled = false;
            return settings;
        }
        const auto object = doc.object();
        settings.enabled = object.value("enabled").toBool(true);
        settings.device = object.value("device").toString(settings.device);
        settings.preferSystemAudio = object.value("prefer_system_audio").toBool(true);
        settings.pulseSink = object.value("pulse_sink").toString(settings.pulseSink);
        settings.volume = std::clamp(object.value("volume").toInt(90), 0, 100);
        const QString directory = object.value("audio_directory").toString("audio");
        settings.audioDirectory = QDir::isAbsolutePath(directory) ? directory
            : QDir(baseDirectory).filePath(directory);
        settings.morning = object.value("morning_start").toInt(5);
        settings.noon = object.value("noon_start").toInt(11);
        settings.afternoon = object.value("afternoon_start").toInt(14);
        settings.evening = object.value("evening_start").toInt(18);
        if (settings.device.isEmpty() || settings.morning < 0
            || settings.morning >= settings.noon || settings.noon >= settings.afternoon
            || settings.afternoon >= settings.evening || settings.evening > 23) {
            qWarning() << "[voice] invalid device or greeting periods; voice disabled";
            settings.enabled = false;
        }
    }
    if (qEnvironmentVariableIsSet("FACEAPP_VOICE_ENABLED"))
        settings.enabled = settings.enabled && qEnvironmentVariableIntValue("FACEAPP_VOICE_ENABLED") != 0;
    return settings;
}

VoiceAnnouncer::VoiceAnnouncer(VoiceSettings settings, QObject *parent)
    : QObject(parent), m_settings(std::move(settings)), m_player(this), m_mixer(this),
      m_timeout(this), m_mixerTimeout(this)
{
    if (!m_settings.preferencesFile.isEmpty()) {
        QSettings saved(m_settings.preferencesFile, QSettings::IniFormat);
        m_settings.volume=std::clamp(saved.value("volume",m_settings.volume).toInt(),0,100);
        m_settings.muted=saved.value("muted",m_settings.muted).toBool();
    }
    m_timeout.setSingleShot(true);
    m_timeout.setInterval(10000);
    connect(&m_timeout, &QTimer::timeout, this, [this] {
        emit playbackFailed(QStringLiteral("audio playback timed out"));
        m_player.kill();
    });
    connect(&m_player, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this](int code, QProcess::ExitStatus status) {
        finishPlayback(code != 0 || status != QProcess::NormalExit,
                       QString::fromLocal8Bit(m_player.readAllStandardError()).trimmed());
    });
    connect(&m_player, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error != QProcess::FailedToStart) return; // Crashes are handled by finished().
        finishPlayback(true, m_player.errorString());
    });
    m_mixerTimeout.setSingleShot(true); m_mixerTimeout.setInterval(5000);
    connect(&m_mixerTimeout,&QTimer::timeout,this,[this] { m_mixer.kill(); });
    const auto mixerDone=[this] {
        m_mixerTimeout.stop();
        QTimer::singleShot(0,this,[this] { applyVolume(); playNext(); });
    };
    connect(&m_mixer,QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),this,
            [this,mixerDone](int code,QProcess::ExitStatus) {
        if (code!=0) emit playbackFailed(QStringLiteral("RK809 volume change failed"));
        mixerDone();
    });
    connect(&m_mixer,&QProcess::errorOccurred,this,[this,mixerDone](QProcess::ProcessError error) {
        if (error!=QProcess::FailedToStart) return;
        emit playbackFailed(m_mixer.errorString()); mixerDone();
    });
    connect(this, &VoiceAnnouncer::playbackFailed, this, [](const QString &message) {
        qWarning().noquote() << "[voice]" << message;
    });
}

VoiceAnnouncer::~VoiceAnnouncer()
{
    // Only shutdown waits, never the camera/recognition frame path.
    disconnect(&m_player, nullptr, this, nullptr);
    disconnect(&m_mixer, nullptr, this, nullptr);
    m_timeout.stop();
    m_mixerTimeout.stop();
    if (m_player.state() != QProcess::NotRunning) {
        m_player.kill();
        m_player.waitForFinished(1000);
    }
    if (m_mixer.state()!=QProcess::NotRunning) { m_mixer.kill(); m_mixer.waitForFinished(1000); }
}

void VoiceAnnouncer::setVolume(int volume, bool muted)
{
    m_settings.volume=std::clamp(volume,0,100); m_settings.muted=muted;
    if (!m_settings.preferencesFile.isEmpty()) {
        QSettings saved(m_settings.preferencesFile,QSettings::IniFormat);
        saved.setValue("volume",m_settings.volume); saved.setValue("muted",muted);
    }
    if (muted || m_settings.volume==0) {
        ++m_playbackGeneration;
        m_pending.clear(); m_playingTrack=-1; m_timeout.stop();
        if (m_player.state()!=QProcess::NotRunning) m_player.kill();
    }
    m_volumeDirty=true; applyVolume();
    emit volumeChanged(m_settings.volume,muted);
}

void VoiceAnnouncer::applyVolume()
{
    if (!m_settings.hardwareVolume) { m_volumeDirty=false; return; }
    if (!m_volumeDirty || m_mixer.state()!=QProcess::NotRunning) return;
    m_volumeDirty=false;
    m_mixer.setProgram(m_settings.mixerProgram);
    m_mixer.setArguments({"-q","-c","rockchiprk809","sset","DAC",
        QString::number(m_settings.muted ? 0 : m_settings.volume)+"%"});
    m_mixer.start(); m_mixerTimeout.start();
}

QString VoiceAnnouncer::phraseForHour(int hour, const VoiceSettings &settings)
{
    if (hour >= settings.morning && hour < settings.noon) return "morning";
    if (hour >= settings.noon && hour < settings.afternoon) return "noon";
    if (hour >= settings.afternoon && hour < settings.evening) return "afternoon";
    return "evening";
}

void VoiceAnnouncer::onFrame(const QVector<FaceObservation> &faces, int hour)
{
    QSet<int> alive;
    for (const auto &face : faces) if (face.trackId >= 0) alive.insert(face.trackId);
    onFrame(faces, hour, alive);
}

void VoiceAnnouncer::onFrame(const QVector<FaceObservation> &faces, int hour,
                            const QSet<int> &aliveTracks)
{
    if (!m_settings.enabled || m_settings.muted || m_settings.volume==0) return;
    m_active.clear();
    m_identities.clear();
    for (const auto &face : faces) {
        if (face.trackId >= 0 && face.identityState != IdentityState::DetectionOnly
            && (face.identityState != IdentityState::Known || !face.personId.isEmpty()))
        {
            m_active.insert(face.trackId);
            m_identities.insert(face.trackId,face.identityState==IdentityState::Known
                ? QStringLiteral("known:")+face.personId : QStringLiteral("unknown"));
        }
    }
    m_announced.intersect(aliveTracks);
    for (auto it = m_pending.begin(); it != m_pending.end();) {
        if (!m_active.contains(it->trackId)) it = m_pending.erase(it);
        else ++it;
    }
    if (m_playingTrack >= 0 && !aliveTracks.contains(m_playingTrack)) {
        ++m_playbackGeneration;
        m_playingTrack = -1;
        m_player.kill();
    }
    for (const auto &face : faces) {
        if (face.trackId < 0 || m_announced.contains(face.trackId)
            || face.identityState == IdentityState::DetectionOnly)
            continue;
        if (face.identityState == IdentityState::Known && face.personId.isEmpty()) continue;
        const QString phrase = face.identityState == IdentityState::Known
            ? phraseForHour(hour, m_settings) : QStringLiteral("denied");
        const auto queued = std::find_if(m_pending.begin(), m_pending.end(),
            [&](const Request &request) { return request.trackId == face.trackId; });
        if (queued != m_pending.end()) {
            queued->phrase = phrase; // Never greet using an obsolete queued decision.
            continue;
        }
        if (m_pending.size() >= 8) continue;
        m_pending.enqueue({face.trackId, phrase});
    }
    playNext();
}

void VoiceAnnouncer::reset()
{
    ++m_playbackGeneration;
    m_announced.clear(); m_active.clear(); m_pending.clear();
    m_identities.clear();
    m_playingTrack = -1;
    m_timeout.stop();
    if (m_player.state() != QProcess::NotRunning) m_player.kill();
}

void VoiceAnnouncer::playNext()
{
    if (!m_settings.enabled || m_settings.muted || m_settings.volume==0) return;
    applyVolume();
    if (m_mixer.state()!=QProcess::NotRunning || m_playingTrack>=0
        || m_player.state() != QProcess::NotRunning) return;
    while (!m_pending.empty()) {
        const auto request = m_pending.dequeue();
        if (!m_active.contains(request.trackId)) continue;
        m_announced.insert(request.trackId);
        const QString file = QDir(m_settings.audioDirectory).filePath(request.phrase + ".wav");
        if (!QFileInfo(file).isFile()) {
            emit playbackFailed(QStringLiteral("missing audio file: ") + file);
            continue;
        }
        m_playingTrack = request.trackId;
        ++m_playbackGeneration;
        m_currentFile = file;
        m_currentIdentity=m_identities.value(request.trackId);
        qInfo().noquote() << "[voice] track=" << request.trackId
                         << "phrase=" << request.phrase << "device=" << m_settings.device;
        emit playbackRequested(request.trackId, request.phrase, file);
        const bool systemAudio = m_settings.preferSystemAudio
            && QFileInfo::exists(m_settings.pulseServer)
            && !QStandardPaths::findExecutable(m_settings.pulseProgram).isEmpty();
        startCurrent(systemAudio);
        return;
    }
}

void VoiceAnnouncer::startCurrent(bool systemAudio)
{
    m_usingPulse=systemAudio;
    m_player.setProgram(systemAudio ? m_settings.pulseProgram : m_settings.playerProgram);
    if (systemAudio) {
        m_player.setArguments({"--server=unix:"+m_settings.pulseServer,
                              "--device="+m_settings.pulseSink,m_currentFile});
    } else m_player.setArguments({"-q","-D",m_settings.device,m_currentFile});
    qInfo() << "[voice] output=" << (systemAudio ? "PulseAudio/RK809" : "ALSA/RK809");
    m_player.start(); m_timeout.start();
}

void VoiceAnnouncer::finishPlayback(bool failed,const QString &message)
{
    m_timeout.stop();
    if (failed && m_usingPulse && m_playingTrack>=0 && m_active.contains(m_playingTrack)) {
        m_usingPulse=false;
        qWarning() << "[voice] system audio unavailable; retrying ALSA";
        const auto generation=m_playbackGeneration;
        const auto track=m_playingTrack;
        const auto identity=m_currentIdentity;
        QTimer::singleShot(0,this,[this,generation,track,identity] {
            if (generation!=m_playbackGeneration || track!=m_playingTrack) return;
            if (m_player.state()!=QProcess::NotRunning) return;
            if (m_active.contains(track) && m_identities.value(track)==identity
                && !m_settings.muted && m_settings.volume>0) startCurrent(false);
            else { m_playingTrack=-1; playNext(); }
        });
        return;
    }
    if (failed && m_playingTrack>=0) emit playbackFailed(message);
    m_playingTrack=-1;
    QTimer::singleShot(0,this,[this] { playNext(); });
}
