#include "opencvfacebackend.h"
#include <QBuffer>
#include <QJsonArray>
#include <QUuid>
#include <algorithm>
#include <stdexcept>

namespace {
int pageArgument(const QJsonObject &payload, const QString &key, int defaultValue, int maximum) {
    if (!payload.contains(key)) return defaultValue;
    bool ok = false; const int value = payload.value(key).toVariant().toInt(&ok);
    if (!ok || value < 1 || value > maximum) throw std::invalid_argument("分页参数错误");
    return value;
}
QJsonObject paginated(const QJsonArray &items, int page, int limit) {
    QJsonArray selected;
    const qint64 start = qint64(page-1)*limit;
    for (qint64 index=start; index<items.size() && index<start+limit; ++index)
        selected.push_back(items.at(int(index)));
    return {{"items", selected}, {"total", items.size()}, {"page", page}, {"limit", limit}};
}
}

void OpenCvFaceBackend::executeAccessCommand(const QString &requestId,
    const QString &operation, const QJsonObject &payload)
{
    try {
        throwIfShutdownRequested(); initializeCore();
        QJsonObject result;
        const std::string personId = payload.value(QStringLiteral("person_id")).toString().toStdString();
        auto personRecord = [this, &personId] {
            if (!facecore::isSafeIdentifier(personId)) throw std::invalid_argument("人员ID格式错误");
            for (const auto &person : m_dataset->loadRegistry()) if (person.personId == personId) return person;
            throw std::invalid_argument("人员不存在");
        };
        if (operation == QStringLiteral("station.configure")) {
            const QString id = payload.value(QStringLiteral("id")).toString();
            const int version = payload.value(QStringLiteral("version")).toInt();
            const QString direction = payload.value(QStringLiteral("direction")).toString();
            const bool enabled = !payload.contains(QStringLiteral("enabled")) || payload.value(QStringLiteral("enabled")).toVariant().toBool();
            if (version < 1 || (direction != "IN" && direction != "OUT")
                || id != "board-in") throw std::invalid_argument("终端配置错误");
            const int currentVersion = m_accessStationVersion;
            if (version < currentVersion) {
                // HTTP configuration and periodic snapshots may arrive out of order.
                result = {{"configured", false}, {"ignored", true}, {"version", currentVersion}};
            } else {
                if (version != m_accessStationVersion || enabled != m_accessStationEnabled) resetTrackingState();
                m_accessStationVersion = version; m_accessStationEnabled = enabled;
                emit stationConfigured(id, direction, version, enabled);
                result = {{"configured", true}};
            }
        } else if (operation == QStringLiteral("people.list")) {
            const QString query = payload.value(QStringLiteral("q")).toString().trimmed();
            std::unordered_map<std::string,int> counts;
            for (const auto &sample : m_dataset->loadManifest()) ++counts[sample.personId];
            QJsonArray items;
            for (const auto &person : m_dataset->loadRegistry()) {
                if (!query.isEmpty() && !utf8(person.personId).contains(query, Qt::CaseInsensitive)
                    && !utf8(person.displayName).contains(query, Qt::CaseInsensitive)) continue;
                items.push_back(QJsonObject{{"person_id", utf8(person.personId)}, {"display_name", utf8(person.displayName)},
                    {"enabled", person.enabled}, {"group", person.group == facecore::PersonGroup::OutGroup ? "out_group" : "in_group"},
                    {"sample_count", counts[person.personId]}});
            }
            result = paginated(items, pageArgument(payload,"page",1,2147483647), pageArgument(payload,"limit",50,200));
        } else if (operation == QStringLiteral("people.create") || operation == QStringLiteral("people.update")) {
            requireDatabaseWritable();
            facecore::PersonRecord record;
            if (operation == "people.create") {
                if (!facecore::isSafeIdentifier(personId)) throw std::invalid_argument("人员ID格式错误");
                for (const auto &existing : m_dataset->loadRegistry())
                    if (existing.personId == personId) throw std::invalid_argument("该人员ID已经存在");
                record.personId = personId; record.enabled = true;
            } else record = personRecord();
            if (payload.contains("display_name")) record.displayName = payload["display_name"].toString().trimmed().toStdString();
            if (record.displayName.empty()) throw std::invalid_argument("姓名不能为空");
            if (payload.contains("group")) {
                const QString group = payload["group"].toString();
                if (group != "in_group" && group != "out_group") throw std::invalid_argument("组别错误");
                record.group = group == "out_group" ? facecore::PersonGroup::OutGroup : facecore::PersonGroup::InGroup;
            }
            if (payload.contains("enabled")) record.enabled = payload["enabled"].toBool();
            if (operation == "people.create") m_dataset->upsertPerson(record); else m_dataset->updatePerson(record);
            finishDatabaseChange(tr("网页人员资料已保存")); result = {{"person_id", utf8(record.personId)}};
        } else if (operation == QStringLiteral("people.delete")) {
            requireDatabaseWritable(); personRecord();
            m_dataset->removePerson(personId); finishDatabaseChange(tr("网页人员记录已归档"));
            result = {{"deleted", true}};
        } else if (operation == QStringLiteral("samples.list")) {
            personRecord(); QJsonArray items;
            for (const auto &sample : m_dataset->loadManifest()) if (sample.personId == personId)
                items.push_back(QJsonObject{{"sample_id", utf8(sample.sampleId)}, {"session_id", utf8(sample.sessionId)},
                    {"source", utf8(sample.source)}, {"quality_score", sample.qualityScore}});
            result = paginated(items, pageArgument(payload,"page",1,2147483647), pageArgument(payload,"limit",50,200));
        } else if (operation == QStringLiteral("samples.read") || operation == QStringLiteral("samples.delete")) {
            personRecord();
            const std::string sampleId = payload.value("sample_id").toString().toStdString();
            if (!facecore::isSafeIdentifier(sampleId)) throw std::invalid_argument("样本ID格式错误");
            const auto records = m_dataset->loadManifest();
            const auto found = std::find_if(records.begin(), records.end(), [&](const auto &sample) {
                return sample.personId == personId && sample.sampleId == sampleId;
            });
            if (found == records.end()) throw std::invalid_argument("样本不存在或不属于该人员");
            if (operation == "samples.delete") {
                requireDatabaseWritable(); m_dataset->removeSample(personId, sampleId);
                finishDatabaseChange(tr("网页样本记录已归档")); result = {{"deleted", true}};
            } else {
                cv::Mat image;
                if (!found->alignedRelativePath.empty()) {
                    try { image = m_dataset->loadAlignedSample(*found); } catch (const std::exception &) {}
                }
                if (image.empty()) image = m_dataset->loadGraySample(*found);
                auto thumbnail = toQImage(image).scaled(256,256,Qt::KeepAspectRatio,Qt::SmoothTransformation);
                QByteArray jpeg; QBuffer output(&jpeg); output.open(QIODevice::WriteOnly);
                if (!thumbnail.save(&output,"JPEG",80)) throw std::runtime_error("无法编码样本照片");
                result = {{"jpeg_base64", QString::fromLatin1(jpeg.toBase64())}};
            }
        } else if (operation == QStringLiteral("samples.add")) {
            requireDatabaseWritable(); personRecord();
            const auto encoded = payload.value("jpeg_base64").toString().toLatin1();
            if (encoded.size() > 12*1024*1024) throw std::invalid_argument("照片过大");
            const auto jpeg = QByteArray::fromBase64(encoded, QByteArray::AbortOnBase64DecodingErrors);
            result = importAccessSample(utf8(personId), jpeg);
        } else if (operation == QStringLiteral("gallery.rebuild")) {
            requireDatabaseWritable(); rebuildGalleryOrThrow();
            setRecognitionEnabled(true);
            result = {{"rebuilt", true}, {"recognition_enabled", m_recognitionEnabled}};
        } else {
            throw std::invalid_argument("该板端操作尚不可用");
        }
        emit accessCommandFinished(requestId, result, {});
    } catch (const std::exception &error) {
        emit accessCommandFinished(requestId, {}, {{"code", "board_operation_failed"}, {"message", utf8(error.what())}});
    }
}
