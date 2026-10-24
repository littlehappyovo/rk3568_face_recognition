#include "voiceannouncer.h"
#include <facecore/tracker.hpp>
#include <QFile>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>

class VoiceTests : public QObject {
    Q_OBJECT
    static FaceObservation face(int id, IdentityState state = IdentityState::Known) {
        FaceObservation item;
        item.trackId = id; item.identityState = state;
        if (state == IdentityState::Known) item.personId = "P1";
        return item;
    }
    VoiceSettings settings(const QString &root, const QString &body = "sleep 0.15\nexit 0\n") {
        VoiceSettings result;
        result.audioDirectory = root;
        result.playerProgram = root + "/player";
        QFile player(result.playerProgram);
        if (!player.open(QIODevice::WriteOnly)) qFatal("cannot write player");
        player.write("#!/bin/sh\n" + body.toUtf8()); player.close();
        player.setPermissions(QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner);
        for (const QString &name : {"morning", "noon", "afternoon", "evening", "denied"}) {
            QFile audio(root + "/" + name + ".wav");
            if (!audio.open(QIODevice::WriteOnly)) qFatal("cannot write audio");
            audio.write("test audio");
        }
        return result;
    }
private slots:
    void periodsRespectBoundaries() {
        QCOMPARE(VoiceAnnouncer::phraseForHour(4), QString("evening"));
        QCOMPARE(VoiceAnnouncer::phraseForHour(5), QString("morning"));
        QCOMPARE(VoiceAnnouncer::phraseForHour(10), QString("morning"));
        QCOMPARE(VoiceAnnouncer::phraseForHour(11), QString("noon"));
        QCOMPARE(VoiceAnnouncer::phraseForHour(13), QString("noon"));
        QCOMPARE(VoiceAnnouncer::phraseForHour(14), QString("afternoon"));
        QCOMPARE(VoiceAnnouncer::phraseForHour(17), QString("afternoon"));
        QCOMPARE(VoiceAnnouncer::phraseForHour(18), QString("evening"));
    }
    void continuousTrackDoesNotRepeatEvenAfterTimeChanges() {
        QTemporaryDir dir; VoiceAnnouncer voice(settings(dir.path()));
        QSignalSpy played(&voice, &VoiceAnnouncer::playbackRequested);
        voice.onFrame({face(7)}, 9);
        QTRY_COMPARE(played.count(), 1);
        QTest::qWait(200);
        for (int i=0; i<30; ++i) voice.onFrame({face(7)}, 20);
        QTest::qWait(50);
        QCOMPARE(played.count(), 1);
    }
    void lostAndReacquiredTargetCanSpeakImmediately() {
        QTemporaryDir dir; VoiceAnnouncer voice(settings(dir.path()));
        QSignalSpy played(&voice, &VoiceAnnouncer::playbackRequested);
        voice.onFrame({face(7)}, 9); QTRY_COMPARE(played.count(), 1);
        voice.onFrame({}, 9);
        voice.onFrame({face(7)}, 12); // Also covers tracker ID reuse after reset.
        QTRY_COMPARE(played.count(), 2);
        QCOMPARE(played.last()[1].toString(), QString("noon"));
    }
    void temporaryOcclusionKeepsTrackAnnouncementUntilRetired() {
        QTemporaryDir dir; VoiceAnnouncer voice(settings(dir.path()));
        QSignalSpy played(&voice, &VoiceAnnouncer::playbackRequested);
        const QSet<int> alive{7};
        voice.onFrame({face(7)}, 9, alive); QTRY_COMPARE(played.count(), 1);
        QTest::qWait(200);
        voice.onFrame({}, 9, alive); // Tracker still owns this temporarily unseen ID.
        voice.onFrame({face(7)}, 9, alive);
        QTest::qWait(200); QCOMPARE(played.count(), 1);
        voice.onFrame({}, 9, {}); // Confirmed loss retires the visit.
        voice.onFrame({face(7)}, 9, alive); QTRY_COMPARE(played.count(), 2);
    }
    void unconfirmedAndLowQualityWaitWithoutRearming() {
        QTemporaryDir dir; VoiceAnnouncer voice(settings(dir.path()));
        QSignalSpy played(&voice, &VoiceAnnouncer::playbackRequested);
        voice.onFrame({face(1, IdentityState::DetectionOnly)}, 9);
        QCOMPARE(played.count(), 0);
        voice.onFrame({face(1)}, 9); QTRY_COMPARE(played.count(), 1);
        voice.onFrame({face(1, IdentityState::DetectionOnly)}, 9);
        voice.onFrame({face(1)}, 9);
        QTest::qWait(200); QCOMPARE(played.count(), 1);
    }
    void unknownDeniedAndGroupsDoNotGateGreeting() {
        QTemporaryDir dir; VoiceAnnouncer voice(settings(dir.path()));
        QSignalSpy played(&voice, &VoiceAnnouncer::playbackRequested);
        auto known = face(2); known.group = QString::fromUtf8("组外");
        voice.onFrame({face(1, IdentityState::Unknown), known}, 16);
        QTRY_COMPARE(played.count(), 2);
        QCOMPARE(played[0][1].toString(), QString("denied"));
        QCOMPARE(played[1][1].toString(), QString("afternoon"));
    }
    void lostQueuedTargetIsNeverPlayedAndLiveInputDoesNotBlock() {
        QTemporaryDir dir; VoiceAnnouncer voice(settings(dir.path(), "sleep 0.4\nexit 0\n"));
        QSignalSpy played(&voice, &VoiceAnnouncer::playbackRequested);
        voice.onFrame({face(1),face(2)}, 9); QTRY_COMPARE(played.count(), 1);
        voice.onFrame({face(1)}, 9); // Target 2 leaves before its turn.
        QTest::qWait(550); QCOMPARE(played.count(), 1);
    }
    void missingPlayerFailsOnceWithoutRepeatedSpawning() {
        QTemporaryDir dir; auto config=settings(dir.path());
        config.playerProgram=dir.path()+"/missing";
        VoiceAnnouncer voice(config);
        QSignalSpy errors(&voice, &VoiceAnnouncer::playbackFailed);
        voice.onFrame({face(1)}, 9); QTRY_COMPARE(errors.count(), 1);
        for (int i=0;i<20;++i) voice.onFrame({face(1)}, 9);
        QTest::qWait(50); QCOMPARE(errors.count(), 1);
    }
    void queuedDecisionUsesLatestStableRecognition() {
        QTemporaryDir dir; VoiceAnnouncer voice(settings(dir.path(), "sleep 0.3\nexit 0\n"));
        QSignalSpy played(&voice, &VoiceAnnouncer::playbackRequested);
        voice.onFrame({face(1),face(2)}, 9); QTRY_COMPARE(played.count(), 1);
        voice.onFrame({face(1),face(2, IdentityState::Unknown)}, 9);
        QTRY_COMPARE(played.count(), 2);
        QCOMPARE(played.last()[1].toString(), QString("denied"));
    }
    void fullQueueStillUpdatesDecisionsAndResetDiscardsPending() {
        QTemporaryDir dir; VoiceAnnouncer voice(settings(dir.path(), "sleep 0.3\nexit 0\n"));
        QSignalSpy played(&voice, &VoiceAnnouncer::playbackRequested);
        QVector<FaceObservation> faces;
        for (int id=1;id<=9;++id) faces.push_back(face(id));
        voice.onFrame(faces,9); QTRY_COMPARE(played.count(),1);
        voice.onFrame(faces,9); // Eight queued targets plus the current player.
        faces[1].identityState=IdentityState::Unknown; faces[1].personId.clear();
        voice.onFrame(faces,9);
        QTRY_COMPARE(played.count(),2);
        QCOMPARE(played.last()[1].toString(),QString("denied"));
        voice.reset();
        QTest::qWait(400); QCOMPARE(played.count(),2);
    }
    void realTrackerRetainsOccludedIdsUntilConfiguredLoss() {
        facecore::TrackerConfig config; config.maximumMissedFrames = 2;
        facecore::FaceTracker tracker(config);
        facecore::FaceDetection detection; detection.box={10,10,100,100};
        const auto first=tracker.update({detection});
        QCOMPARE(first.size(), std::size_t(1));
        const auto id=first[0].trackId;
        QVERIFY(tracker.update({}).empty());
        QVERIFY(tracker.update({}).empty());
        QCOMPARE(tracker.activeTrackIds(), std::vector<std::uint64_t>{id});
        const auto recovered=tracker.update({detection});
        QCOMPARE(recovered[0].trackId, id);
        tracker.update({}); tracker.update({}); tracker.update({});
        QVERIFY(tracker.activeTrackIds().empty());
        QVERIFY(tracker.update({detection})[0].trackId != id);
        tracker.reset(); QVERIFY(tracker.activeTrackIds().empty());
    }
    void invalidConfigurationDisablesVoiceAndDisabledVoiceNeverSpawns() {
        QTemporaryDir dir; QVERIFY(QDir(dir.path()).mkdir("config"));
        QFile file(dir.path()+"/config/voice.json"); QVERIFY(file.open(QIODevice::WriteOnly));
        file.write("{\"morning_start\":14,\"noon_start\":11}"); file.close();
        QVERIFY(!VoiceSettings::load(dir.path()).enabled);
        auto config=settings(dir.path()); config.enabled=false;
        VoiceAnnouncer voice(config);
        QSignalSpy played(&voice, &VoiceAnnouncer::playbackRequested);
        voice.onFrame({face(1)},9); QTest::qWait(50); QCOMPARE(played.count(),0);
    }
    void volumeAndMutePersistAndPreventPlayback() {
        QTemporaryDir dir; auto config=settings(dir.path());
        config.preferencesFile=dir.path()+"/voice.ini";
        {
            VoiceAnnouncer voice(config);
            QSignalSpy played(&voice,&VoiceAnnouncer::playbackRequested);
            QSignalSpy changed(&voice,&VoiceAnnouncer::volumeChanged);
            voice.setVolume(73,true);
            QCOMPARE(voice.volume(),73); QVERIFY(voice.muted());
            QCOMPARE(changed.count(),1);
            voice.onFrame({face(1)},9); QTest::qWait(50); QCOMPARE(played.count(),0);
        }
        VoiceAnnouncer reopened(config);
        QCOMPARE(reopened.volume(),73); QVERIFY(reopened.muted());
        reopened.setVolume(120,false); QCOMPARE(reopened.volume(),100);
    }
    void systemAudioFailureFallsBackToAlsaOncePerVisit() {
        QTemporaryDir dir; auto config=settings(dir.path());
        config.preferSystemAudio=true;
        config.pulseServer=dir.path()+"/pulse-socket";
        config.pulseProgram=dir.path()+"/pulse-player";
        QFile server(config.pulseServer); QVERIFY(server.open(QIODevice::WriteOnly)); server.close();
        QFile pulse(config.pulseProgram); QVERIFY(pulse.open(QIODevice::WriteOnly));
        pulse.write("#!/bin/sh\ntouch '"+dir.path().toUtf8()+"/pulse-used'\nexit 1\n"); pulse.close();
        QVERIFY(pulse.setPermissions(QFile::ReadOwner|QFile::WriteOwner|QFile::ExeOwner));
        QFile player(config.playerProgram); QVERIFY(player.open(QIODevice::WriteOnly));
        player.write("#!/bin/sh\nprintf '%s\\n' \"$@\" > '"+dir.path().toUtf8()+"/alsa-used'\nexit 0\n");
        player.close();
        VoiceAnnouncer voice(config);
        QSignalSpy played(&voice,&VoiceAnnouncer::playbackRequested);
        voice.onFrame({face(1)},9);
        QTRY_VERIFY(QFile::exists(dir.path()+"/pulse-used"));
        QTRY_VERIFY(QFile::exists(dir.path()+"/alsa-used"));
        QCOMPARE(played.count(),1);
    }
    void failedVolumeMixerDoesNotBlockAnnouncements() {
        QTemporaryDir dir; auto config=settings(dir.path());
        config.hardwareVolume=true; config.mixerProgram=dir.path()+"/missing-mixer";
        VoiceAnnouncer voice(config);
        QSignalSpy played(&voice,&VoiceAnnouncer::playbackRequested);
        QSignalSpy errors(&voice,&VoiceAnnouncer::playbackFailed);
        voice.onFrame({face(1)},9);
        QTRY_COMPARE(played.count(),1); QCOMPARE(errors.count(),1);
        voice.setVolume(55,true);
        voice.onFrame({face(2)},9); QTest::qWait(200); QCOMPARE(played.count(),1);
    }
    void deferredFallbackCancelsWhenRecognitionBecomesUnconfirmed_data() {
        QTest::addColumn<QString>("change");
        QTest::newRow("unconfirmed") << QString("unconfirmed");
        QTest::newRow("unknown") << QString("unknown");
        QTest::newRow("reset") << QString("reset");
        QTest::newRow("mute") << QString("mute");
        QTest::newRow("new-request") << QString("new-request");
    }
    void deferredFallbackCancelsWhenRecognitionBecomesUnconfirmed() {
        QFETCH(QString,change);
        QTemporaryDir dir; auto config=settings(dir.path());
        config.preferSystemAudio=true;
        config.pulseServer=dir.path()+"/pulse-socket";
        config.pulseProgram=dir.path()+"/pulse-player";
        QFile server(config.pulseServer); QVERIFY(server.open(QIODevice::WriteOnly)); server.close();
        QFile pulse(config.pulseProgram); QVERIFY(pulse.open(QIODevice::WriteOnly));
        pulse.write("#!/bin/sh\nexit 1\n"); pulse.close();
        QVERIFY(pulse.setPermissions(QFile::ReadOwner|QFile::WriteOwner|QFile::ExeOwner));
        QFile player(config.playerProgram); QVERIFY(player.open(QIODevice::WriteOnly));
        player.write("#!/bin/sh\nprintf '%s\\n' \"$4\" >> '"+dir.path().toUtf8()+"/stale-alsa'\nexit 0\n"); player.close();
        VoiceAnnouncer voice(config);
        voice.onFrame({face(1)},9);
        bool pulseFinished=false;
        for (auto *process:voice.findChildren<QProcess *>()) {
            if (process->program()!=config.pulseProgram) continue;
            connect(process,QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),&voice,
                    [&] {
                if (pulseFinished) return;
                pulseFinished=true;
                if (change=="reset" || change=="new-request") voice.reset();
                else if (change=="mute") voice.setVolume(90,true);
                else voice.onFrame({face(1,change=="unknown" ? IdentityState::Unknown : IdentityState::DetectionOnly)},9);
                if (change=="new-request") voice.onFrame({face(2)},12);
            });
        }
        QTRY_VERIFY(pulseFinished); QTest::qWait(100);
        if (change=="new-request") {
            QTRY_VERIFY(QFile::exists(dir.path()+"/stale-alsa"));
            QFile used(dir.path()+"/stale-alsa"); QVERIFY(used.open(QIODevice::ReadOnly));
            const auto lines=used.readAll().trimmed().split('\n');
            QCOMPARE(lines.size(),1); QVERIFY(lines[0].endsWith("noon.wav"));
        } else QVERIFY(!QFile::exists(dir.path()+"/stale-alsa"));
    }
    void liveRk809Playback() {
        if (qEnvironmentVariableIntValue("FACEAPP_LIVE_AUDIO_TEST")!=1)
            QSKIP("Set FACEAPP_LIVE_AUDIO_TEST=1 on the RK3568 to play real prompts");
        auto config=VoiceSettings::load(QString::fromUtf8(FACE_VOICE_PROJECT_ROOT));
        QVERIFY(config.enabled);
        VoiceAnnouncer voice(config);
        QSignalSpy played(&voice,&VoiceAnnouncer::playbackRequested);
        QSignalSpy errors(&voice,&VoiceAnnouncer::playbackFailed);
        voice.onFrame({face(1)},9); QTRY_COMPARE(played.count(),1);
        QTest::qWait(2200);
        voice.onFrame({face(1)},18); QTest::qWait(100); QCOMPARE(played.count(),1);
        voice.onFrame({},18);
        voice.onFrame({face(2,IdentityState::Unknown)},18); QTRY_COMPARE(played.count(),2);
        QTest::qWait(3500); QCOMPARE(errors.count(),0);
    }
};
QTEST_GUILESS_MAIN(VoiceTests)
#include "voice_tests.moc"
