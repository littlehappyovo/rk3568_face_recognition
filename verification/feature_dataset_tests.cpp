#include <facecore/dataset.hpp>
#include <QTemporaryDir>
#include <QtTest>
#include <fstream>

class DatasetManagementTests : public QObject {
    Q_OBJECT
    static facecore::SampleRecord add(facecore::DatasetStore &store,
                                      const std::string &person, const std::string &id) {
        facecore::SampleRecord sample;
        sample.personId = person;
        sample.sampleId = id;
        sample.sessionId = "SESSION1";
        sample.preprocessingVersion = "gray128-eq-v1";
        return store.saveSample(cv::Mat(128, 128, CV_8UC1, cv::Scalar(100)),
                                cv::Mat(112, 112, CV_8UC3, cv::Scalar(40, 80, 120)), sample);
    }
private slots:
    void editPreservesSamplesAndSurvivesReopen() {
        QTemporaryDir temp;
        facecore::DatasetStore store(temp.path().toStdString());
        store.upsertPerson({"P1", "Old", facecore::PersonGroup::InGroup, true});
        const auto sample = add(store, "P1", "S1");
        store.updatePerson({"P1", "王,\"小明\"", facecore::PersonGroup::OutGroup, false});
        facecore::DatasetStore reopened(temp.path().toStdString());
        const auto people = reopened.loadRegistry();
        QCOMPARE(people.size(), std::size_t(1));
        QCOMPARE(people[0].displayName, std::string("王,\"小明\""));
        QVERIFY(!people[0].enabled);
        QVERIFY(people[0].group == facecore::PersonGroup::OutGroup);
        QCOMPARE(reopened.loadManifest().size(), std::size_t(1));
        QVERIFY(!reopened.loadAlignedSample(sample).empty());
        QVERIFY(std::filesystem::exists(store.root() / ".gallery-stale-mobilefacenet_rknn"));
    }
    void deletingPersonDoesNotDeleteOtherPeopleOrReuseOldSamples() {
        QTemporaryDir temp;
        facecore::DatasetStore store(temp.path().toStdString());
        store.upsertPerson({"P1", "One", facecore::PersonGroup::InGroup, true});
        store.upsertPerson({"P2", "Two", facecore::PersonGroup::OutGroup, true});
        const auto removed = add(store, "P1", "S1");
        add(store, "P2", "S2");
        store.removePerson("P1");
        QCOMPARE(store.loadRegistry().size(), std::size_t(1));
        QCOMPARE(store.loadRegistry()[0].personId, std::string("P2"));
        QCOMPARE(store.loadManifest().size(), std::size_t(1));
        QCOMPARE(store.loadManifest()[0].personId, std::string("P2"));
        QVERIFY(std::filesystem::exists(store.root() / removed.relativePath));
        QVERIFY(!std::filesystem::is_empty(store.root() / "archive"));
        store.upsertPerson({"P1", "New person", facecore::PersonGroup::InGroup, true});
        QCOMPARE(store.loadManifest().size(), std::size_t(1));
        QVERIFY_EXCEPTION_THROWN(store.removePerson("../P2"), std::invalid_argument);
    }
    void sampleDeletionRequiresTheCorrectOwner() {
        QTemporaryDir temp;
        facecore::DatasetStore store(temp.path().toStdString());
        store.upsertPerson({"P1", "One", facecore::PersonGroup::InGroup, true});
        store.upsertPerson({"P2", "Two", facecore::PersonGroup::InGroup, true});
        add(store, "P1", "S1");
        add(store, "P1", "S2");
        QVERIFY_EXCEPTION_THROWN(store.removeSample("P2", "S1"), std::invalid_argument);
        QCOMPARE(store.loadManifest().size(), std::size_t(2));
        store.removeSample("P1", "S1");
        QCOMPARE(store.loadManifest().size(), std::size_t(1));
        QCOMPARE(store.loadManifest()[0].sampleId, std::string("S2"));
        QCOMPARE(store.loadRegistry().size(), std::size_t(2));
    }
    void editingCannotCreatePeopleOrBypassCaptureIdentityProtection() {
        QTemporaryDir temp;
        facecore::DatasetStore store(temp.path().toStdString());
        store.upsertPerson({"P1", "One", facecore::PersonGroup::InGroup, true});
        QVERIFY_EXCEPTION_THROWN(store.updatePerson({"MISSING", "No", facecore::PersonGroup::InGroup, true}), std::invalid_argument);
        QVERIFY_EXCEPTION_THROWN(store.updatePerson({"P1", "Bad\nName", facecore::PersonGroup::InGroup, true}), std::invalid_argument);
        QVERIFY_EXCEPTION_THROWN(store.upsertPerson({"P1", "Other", facecore::PersonGroup::InGroup, true}), std::invalid_argument);
        QCOMPARE(store.loadRegistry()[0].displayName, std::string("One"));
    }
};
QTEST_GUILESS_MAIN(DatasetManagementTests)
#include "feature_dataset_tests.moc"
