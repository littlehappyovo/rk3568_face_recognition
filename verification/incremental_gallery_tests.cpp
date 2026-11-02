#ifndef Q_MOC_RUN
#include "facecore/embedding.hpp"
#include "facecore/recognizer.hpp"
#include <QFile>
#include <QTemporaryDir>
#include <QtTest>
#include <opencv2/core.hpp>
#include <memory>
#endif
#include <QtTest>

#ifndef Q_MOC_RUN
namespace {
class CountingExtractor final : public facecore::IEmbeddingExtractor {
public:
    mutable int calls = 0;
    cv::Mat extract(const cv::Mat& face) const override {
        ++calls;
        if (face.empty()) throw std::runtime_error("empty input");
        const auto color = face.at<cv::Vec3b>(0, 0);
        return facecore::normalizeEmbedding((cv::Mat_<float>(1, 3) << color[0], color[1], color[2]));
    }
    std::string name() const override { return "test-counting-extractor"; }
    std::string inputContract() const override { return "rgb-test-v1"; }
};

facecore::LabeledFaceSample sample(const std::string& id, const std::string& person,
                                  cv::Scalar color, unsigned seed) {
    facecore::LabeledFaceSample result;
    result.sampleId = id;
    result.personId = person;
    result.sessionId = "session";
    result.face.grayFace = cv::Mat(32,32,CV_8UC1);
    cv::RNG rng(seed);
    rng.fill(result.face.grayFace, cv::RNG::UNIFORM, 0, 256);
    result.face.alignedBgrFace = cv::Mat(2,2,CV_8UC3,color);
    result.grayFingerprint = "gray-" + std::to_string(seed);
    result.alignedFingerprint = "color-" + std::to_string(seed);
    return result;
}

facecore::SFaceConfig embeddingConfig() {
    facecore::SFaceConfig config;
    config.embeddingDim = 3;
    config.modelIdentity = "identity-v1";
    config.preprocessingVersion = "aligned-v1";
    config.cosineThreshold = .6;
    config.minimumCosineMargin = .04;
    return config;
}
}
#endif

class IncrementalGalleryTests : public QObject {
    Q_OBJECT
private slots:
    void embeddingReloadAppendChangeAndRemoval() {
        QTemporaryDir dir; QVERIFY(dir.isValid());
        const auto path = std::filesystem::path(dir.path().toStdString()) / "embedding.yml";
        auto extractor = std::make_shared<CountingExtractor>();
        facecore::EmbeddingFaceRecognizer initial(extractor, embeddingConfig());
        auto a = sample("a", "P1", cv::Scalar(100,0,0), 1);
        auto b = sample("b", "P1", cv::Scalar(0,100,0), 2);
        auto c = sample("c", "P2", cv::Scalar(0,0,100), 3);
        auto first = initial.buildGalleryIncremental({a,b}, path, "model-sha-1");
        QCOMPARE(first.recomputed, std::size_t(2)); QCOMPARE(first.reused, std::size_t(0));
        QCOMPARE(extractor->calls, 2);
        facecore::EmbeddingFaceRecognizer restarted(extractor, embeddingConfig());
        auto second = restarted.buildGalleryIncremental({a,b,c}, path, "model-sha-1");
        QCOMPARE(second.recomputed, std::size_t(1)); QCOMPARE(second.reused, std::size_t(2));
        QCOMPARE(extractor->calls, 3);
        facecore::EmbeddingFaceRecognizer full(extractor, embeddingConfig());
        full.buildGallery({a,b,c});
        auto incResult = restarted.recognize(a.face);
        auto fullResult = full.recognize(a.face);
        QCOMPARE(incResult.personId, fullResult.personId);
        QVERIFY(std::abs(incResult.bestScore - fullResult.bestScore) < 1e-5);
        b = sample("b", "P1", cv::Scalar(100,100,0), 9);
        auto changed = restarted.buildGalleryIncremental({a,b,c}, path, "model-sha-1");
        QCOMPARE(changed.recomputed, std::size_t(1)); QCOMPARE(changed.reused, std::size_t(2));
        auto removed = restarted.buildGalleryIncremental({b,c}, path, "model-sha-1");
        QCOMPARE(removed.recomputed, std::size_t(0)); QCOMPARE(removed.reused, std::size_t(2));
        QCOMPARE(removed.removed, std::size_t(1));
        auto newModel = restarted.buildGalleryIncremental({b,c}, path, "model-sha-2");
        QCOMPARE(newModel.recomputed, std::size_t(2)); QCOMPARE(newModel.reused, std::size_t(0));
    }
    void lbphReloadAppendAndDeletePreservesRecognition() {
        QTemporaryDir dir; QVERIFY(dir.isValid());
        const auto path = std::filesystem::path(dir.path().toStdString()) / "lbph.yml";
        facecore::LbphConfig config; config.preprocessingVersion = "gray-v1";
        auto a = sample("a", "P1", cv::Scalar(100,0,0), 11);
        auto b = sample("b", "P2", cv::Scalar(0,100,0), 22);
        auto c = sample("c", "P1", cv::Scalar(0,0,100), 33);
        facecore::LbphFaceRecognizer first(config);
        auto initial = first.trainIncremental({a,b}, path);
        QCOMPARE(initial.recomputed, std::size_t(2));
        facecore::LbphFaceRecognizer restarted(config);
        auto next = restarted.trainIncremental({a,b,c}, path);
        QCOMPARE(next.recomputed, std::size_t(1)); QCOMPARE(next.reused, std::size_t(2));
        facecore::LbphFaceRecognizer full(config); full.train({a,b,c});
        auto actual = restarted.recognize(b.face), expected = full.recognize(b.face);
        QCOMPARE(actual.personId, expected.personId);
        QVERIFY(std::abs(actual.bestScore - expected.bestScore) < 1e-5);
        auto removed = restarted.trainIncremental({a,c}, path);
        QCOMPARE(removed.recomputed, std::size_t(0)); QCOMPARE(removed.reused, std::size_t(2));
        QCOMPARE(removed.removed, std::size_t(1));
        const auto deleted = restarted.recognize(b.face);
        QVERIFY2(deleted.personId != "P2", "Deleted identity must not survive in the rebuilt gallery");
    }
    void corruptCacheIsRecomputedAndCannotChangeIdentity() {
        QTemporaryDir dir; QVERIFY(dir.isValid());
        const auto path = std::filesystem::path(dir.path().toStdString()) / "cache.yml";
        auto extractor = std::make_shared<CountingExtractor>();
        facecore::EmbeddingFaceRecognizer recognizer(extractor, embeddingConfig());
        auto a = sample("a", "P1", cv::Scalar(100,0,0), 11);
        recognizer.buildGalleryIncremental({a}, path, "model-sha-1");
        QFile broken(QString::fromStdString(path.string()));
        QVERIFY(broken.open(QIODevice::WriteOnly | QIODevice::Truncate));
        QCOMPARE(broken.write("broken"), qint64(6)); broken.close();
        auto retried = recognizer.buildGalleryIncremental({a}, path, "model-sha-1");
        QCOMPARE(retried.recomputed, std::size_t(1)); QCOMPARE(retried.reused, std::size_t(0));
        QCOMPARE(QString::fromStdString(recognizer.recognize(a.face).personId), QString("P1"));
    }
    void sampleIdReassignedToAnotherPersonIsNotReused() {
        QTemporaryDir dir; QVERIFY(dir.isValid());
        const auto path = std::filesystem::path(dir.path().toStdString()) / "cache.yml";
        auto extractor = std::make_shared<CountingExtractor>();
        facecore::EmbeddingFaceRecognizer recognizer(extractor, embeddingConfig());
        auto a = sample("a", "P1", cv::Scalar(100,0,0), 11);
        recognizer.buildGalleryIncremental({a}, path, "model-sha-1");
        a.personId = "P2";
        const auto changed = recognizer.buildGalleryIncremental({a}, path, "model-sha-1");
        QCOMPARE(changed.recomputed, std::size_t(1));
        QCOMPARE(changed.reused, std::size_t(0));
        QCOMPARE(QString::fromStdString(recognizer.recognize(a.face).personId), QString("P2"));
    }
    void failedExtractionKeepsPreviousGalleryAndCache() {
        QTemporaryDir dir; QVERIFY(dir.isValid());
        const auto path = std::filesystem::path(dir.path().toStdString()) / "cache.yml";
        auto extractor = std::make_shared<CountingExtractor>();
        facecore::EmbeddingFaceRecognizer recognizer(extractor, embeddingConfig());
        auto a = sample("a", "P1", cv::Scalar(100,0,0), 11);
        auto b = sample("b", "P2", cv::Scalar(0,100,0), 22);
        recognizer.buildGalleryIncremental({a,b}, path, "model-sha-1");
        QFile cache(QString::fromStdString(path.string()));
        QVERIFY(cache.open(QIODevice::ReadOnly)); const auto previous = cache.readAll(); cache.close();
        auto bad = sample("c", "P3", cv::Scalar(0,0,0), 33);
        QVERIFY_EXCEPTION_THROWN(recognizer.buildGalleryIncremental({a,b,bad}, path, "model-sha-1"), std::runtime_error);
        QVERIFY(cache.open(QIODevice::ReadOnly)); QCOMPARE(cache.readAll(), previous); cache.close();
        QCOMPARE(QString::fromStdString(recognizer.recognize(b.face).personId), QString("P2"));
    }
    void savedGalleryCannotLoadAgainstDifferentModelBytes() {
        QTemporaryDir dir; QVERIFY(dir.isValid());
        const auto root = std::filesystem::path(dir.path().toStdString());
        auto extractor = std::make_shared<CountingExtractor>();
        auto a = sample("a", "P1", cv::Scalar(100,0,0), 11);
        facecore::EmbeddingFaceRecognizer first(extractor, embeddingConfig());
        first.buildGalleryIncremental({a}, root / "cache-a.yml", "model-sha-1");
        first.save(root / "gallery.yml");
        facecore::EmbeddingFaceRecognizer sameModel(extractor, embeddingConfig());
        sameModel.buildGalleryIncremental({a}, root / "cache-b.yml", "model-sha-1");
        sameModel.load(root / "gallery.yml");
        QCOMPARE(QString::fromStdString(sameModel.recognize(a.face).personId), QString("P1"));
        facecore::EmbeddingFaceRecognizer changedModel(extractor, embeddingConfig());
        changedModel.buildGalleryIncremental({a}, root / "cache-c.yml", "model-sha-2");
        QVERIFY_EXCEPTION_THROWN(changedModel.load(root / "gallery.yml"), std::runtime_error);
    }
    void legacyGalleryMustBeRebuiltEvenWhenIdentityDeclaresHash() {
        QTemporaryDir dir; QVERIFY(dir.isValid());
        const auto path = std::filesystem::path(dir.path().toStdString()) / "legacy.yml";
        auto extractor = std::make_shared<CountingExtractor>();
        auto a = sample("a", "P1", cv::Scalar(100,0,0), 11);
        auto config = embeddingConfig();
        config.modelIdentity = "test-model-sha256-" + std::string(64, 'a');
        facecore::EmbeddingGallery legacy;
        legacy.build({a}, *extractor);
        legacy.save(path, .6, .04, config.modelIdentity, config.preprocessingVersion + ":rgb-test-v1");
        facecore::EmbeddingFaceRecognizer good(extractor, config);
        good.setModelFingerprint(std::string(64, 'a'));
        QVERIFY_EXCEPTION_THROWN(good.load(path), std::runtime_error);
        good.setModelFingerprint(std::string(64, 'b'));
        QCOMPARE(good.identityCount(), std::size_t(0));
        QVERIFY_EXCEPTION_THROWN(good.load(path), std::runtime_error);
        config.modelIdentity = "identity-without-hash";
        legacy.save(path, .6, .04, config.modelIdentity, config.preprocessingVersion + ":rgb-test-v1");
        facecore::EmbeddingFaceRecognizer unverified(extractor, config);
        unverified.setModelFingerprint(std::string(64, 'a'));
        QVERIFY_EXCEPTION_THROWN(unverified.load(path), std::runtime_error);
    }
    void galleryRejectsChangedExtractorConfiguration() {
        QTemporaryDir dir; QVERIFY(dir.isValid());
        const auto root = std::filesystem::path(dir.path().toStdString());
        auto extractor = std::make_shared<CountingExtractor>();
        auto a = sample("a", "P1", cv::Scalar(100,0,0), 11);
        auto config = embeddingConfig();
        facecore::EmbeddingFaceRecognizer first(extractor, config);
        first.buildGalleryIncremental({a}, root / "cache.yml", "model-sha-1");
        first.save(root / "gallery.yml");
        config.rknn.inputColorOrder = "bgr";
        config.rknn.inputScale = 0.25;
        facecore::EmbeddingFaceRecognizer changed(extractor, config);
        changed.setModelFingerprint("model-sha-1");
        QVERIFY_EXCEPTION_THROWN(changed.load(root / "gallery.yml"), std::runtime_error);
    }
};
QTEST_APPLESS_MAIN(IncrementalGalleryTests)
#include "incremental_gallery_tests.moc"
