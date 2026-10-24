#include "facedatabasedialog.h"
#include "facebackendservice.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QScreen>
#include <QShowEvent>
#include <QSignalBlocker>
#include <QSplitter>
#include <QTableWidget>
#include <QVBoxLayout>
#include <algorithm>

FaceDatabaseDialog::FaceDatabaseDialog(FaceBackendService *backend, QWidget *parent)
    : QDialog(parent)
{
    setObjectName(QStringLiteral("faceDatabaseDialog"));
    setWindowTitle(tr("人脸数据库管理"));
    setWindowModality(Qt::WindowModal);
    resize(1240, 800);
    auto *layout = new QVBoxLayout(this);
    auto *toolbar = new QHBoxLayout;
    m_search = new QLineEdit(this);
    m_search->setObjectName(QStringLiteral("databaseSearch"));
    m_search->setPlaceholderText(tr("搜索人员ID或姓名"));
    m_refresh = new QPushButton(tr("刷新人脸库"), this);
    toolbar->addWidget(m_search, 1);
    toolbar->addWidget(m_refresh);
    layout->addLayout(toolbar);
    m_summary = new QLabel(this);
    layout->addWidget(m_summary);

    auto *splitter = new QSplitter(Qt::Horizontal, this);
    m_table = new QTableWidget(0, 5, splitter);
    m_table->setObjectName(QStringLiteral("databasePeople"));
    m_table->setHorizontalHeaderLabels({tr("人员ID"), tr("姓名"), tr("组别"), tr("样本"), tr("状态")});
    m_table->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_table->setSelectionMode(QAbstractItemView::SingleSelection);
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_table->verticalHeader()->setVisible(false);
    m_table->verticalHeader()->setDefaultSectionSize(44);
    m_table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    splitter->addWidget(m_table);

    auto *details = new QWidget(splitter);
    auto *detailLayout = new QVBoxLayout(details);
    auto *form = new QFormLayout;
    m_id = new QLineEdit(details);
    m_id->setReadOnly(true);
    m_name = new QLineEdit(details);
    m_name->setObjectName(QStringLiteral("databasePersonName"));
    m_name->setMaxLength(128);
    m_group = new QComboBox(details);
    m_group->addItem(tr("组内"), QStringLiteral("in"));
    m_group->addItem(tr("组外"), QStringLiteral("out"));
    m_enabled = new QCheckBox(tr("参与人脸识别"), details);
    m_enabled->setObjectName(QStringLiteral("databasePersonEnabled"));
    form->addRow(tr("人员ID："), m_id);
    form->addRow(tr("姓名："), m_name);
    form->addRow(tr("组别："), m_group);
    form->addRow(m_enabled);
    detailLayout->addLayout(form);
    auto *actions = new QHBoxLayout;
    m_save = new QPushButton(tr("保存资料"), details);
    m_save->setObjectName(QStringLiteral("databaseSavePerson"));
    m_deletePerson = new QPushButton(tr("删除人员…"), details);
    m_capture = new QPushButton(tr("为此人采集"), details);
    actions->addWidget(m_save);
    actions->addWidget(m_deletePerson);
    actions->addWidget(m_capture);
    detailLayout->addLayout(actions);
    detailLayout->addWidget(new QLabel(tr("人脸样本（选择照片可查看采集批次与质量）"), details));
    m_samples = new QListWidget(details);
    m_samples->setObjectName(QStringLiteral("databaseSamples"));
    m_samples->setViewMode(QListView::IconMode);
    m_samples->setResizeMode(QListView::Adjust);
    m_samples->setMovement(QListView::Static);
    m_samples->setIconSize(QSize(112, 112));
    m_samples->setGridSize(QSize(144, 160));
    m_samples->setSpacing(6);
    detailLayout->addWidget(m_samples, 1);
    m_deleteSample = new QPushButton(tr("删除选中样本…"), details);
    detailLayout->addWidget(m_deleteSample);
    splitter->addWidget(details);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 1);
    layout->addWidget(splitter, 1);
    m_message = new QLabel(tr("新增人员请在主界面填写资料并采集，或导入照片。"), this);
    m_message->setWordWrap(true);
    layout->addWidget(m_message);
    auto *close = new QPushButton(tr("返回识别界面"), this);
    layout->addWidget(close);
    // The board reports ~472 DPI. Keep dialog spacing in pixels, as in MainWindow.
    for (auto *childLayout : findChildren<QLayout *>()) {
        childLayout->setSpacing(8);
        childLayout->setContentsMargins(10, 8, 10, 8);
    }
    m_table->horizontalHeader()->setFixedHeight(40);
    setEditingEnabled(false);

    connect(close, &QPushButton::clicked, this, &QDialog::accept);
    connect(m_refresh, &QPushButton::clicked, this, [this] {
        setPending();
        emit refreshRequested();
    });
    connect(m_search, &QLineEdit::textChanged, this, &FaceDatabaseDialog::filterPeople);
    connect(m_table, &QTableWidget::itemSelectionChanged, this, &FaceDatabaseDialog::selectPerson);
    connect(m_samples, &QListWidget::currentItemChanged, this, [this](QListWidgetItem *item) {
        m_deleteSample->setEnabled(item != nullptr);
        if (item) m_message->setText(item->toolTip());
    });
    connect(m_save, &QPushButton::clicked, this, [this] {
        if (m_selectedId.isEmpty()) return;
        const PersonProfile profile{m_selectedId, m_name->text().trimmed(), m_group->currentData().toString()};
        if (profile.displayName.isEmpty()) {
            m_message->setText(tr("姓名不能为空。"));
            return;
        }
        const bool enabled = m_enabled->isChecked();
        setPending();
        emit updatePersonRequested(profile, enabled);
    });
    connect(m_deletePerson, &QPushButton::clicked, this, [this] {
        if (m_selectedId.isEmpty()) return;
        const QString id = m_selectedId;
        if (QMessageBox::question(this, tr("删除人员"),
                tr("将%1［%2］及其样本移出当前人脸库？\n登记和样本索引会归档，照片保留。删除后需重建模型。")
                    .arg(m_name->text(), id),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No) != QMessageBox::Yes) return;
        setPending();
        emit deletePersonRequested(id);
    });
    connect(m_deleteSample, &QPushButton::clicked, this, [this] {
        const auto *item = m_samples->currentItem();
        if (!item || m_selectedId.isEmpty()) return;
        // The confirmation dialog processes queued previews, which may delete item
        // or change the selected person. Keep only value copies across that wait.
        const QString sampleId = item->data(Qt::UserRole).toString();
        const QString id = m_selectedId;
        if (QMessageBox::question(this, tr("删除样本"),
                tr("将这张样本移出人脸库？\n样本索引会归档，照片保留。删除后需重建模型。"),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No) != QMessageBox::Yes) return;
        setPending();
        emit deleteSampleRequested(id, sampleId);
    });
    connect(m_capture, &QPushButton::clicked, this, [this] {
        // Use the saved registry, rather than unsaved edits, to label new samples.
        for (const auto &person : m_people) {
            if (person.profile.personId != m_selectedId) continue;
            if (!person.enabled) { m_message->setText(tr("请先启用此人员并保存。")); return; }
            emit capturePersonSelected(person.profile);
            accept();
            return;
        }
    });
    connect(this, &FaceDatabaseDialog::refreshRequested, backend,
            &FaceBackendService::refreshFaceDatabase, Qt::QueuedConnection);
    connect(this, &FaceDatabaseDialog::samplesRequested, backend,
            &FaceBackendService::loadPersonSamples, Qt::QueuedConnection);
    connect(this, &FaceDatabaseDialog::updatePersonRequested, backend,
            &FaceBackendService::updateDatabasePerson, Qt::QueuedConnection);
    connect(this, &FaceDatabaseDialog::deletePersonRequested, backend,
            &FaceBackendService::deleteDatabasePerson, Qt::QueuedConnection);
    connect(this, &FaceDatabaseDialog::deleteSampleRequested, backend,
            &FaceBackendService::deleteDatabaseSample, Qt::QueuedConnection);
    connect(backend, &FaceBackendService::faceDatabaseChanged, this, &FaceDatabaseDialog::setPeople);
    connect(backend, &FaceBackendService::personSamplesChanged, this, &FaceDatabaseDialog::setSamples);
    connect(backend, &FaceBackendService::databaseOperationFinished, this, [this](const QString &message) {
        m_message->setText(message);
    });
    connect(backend, &FaceBackendService::errorOccurred, this, [this] {
        m_refresh->setEnabled(true);
        m_table->setEnabled(true);
        setEditingEnabled(!m_selectedId.isEmpty());
    });
}

void FaceDatabaseDialog::showEvent(QShowEvent *event) {
    QDialog::showEvent(event);
    if (screen()) resize(size().boundedTo(screen()->availableGeometry().size() - QSize(40, 60)));
    setPending();
    emit refreshRequested();
}

void FaceDatabaseDialog::setEditingEnabled(bool enabled) {
    for (QWidget *widget : QList<QWidget *>{m_name, m_group, m_enabled,
                            m_save, m_deletePerson, m_capture, m_samples}) widget->setEnabled(enabled);
    m_deleteSample->setEnabled(enabled && m_samples->currentItem());
}

void FaceDatabaseDialog::setPending() {
    setEditingEnabled(false);
    m_refresh->setEnabled(false);
    m_table->setEnabled(false);
    m_message->setText(tr("正在读取或保存人脸库…"));
}

void FaceDatabaseDialog::setPeople(const FaceDatabasePeople &people) {
    const auto previousId = m_selectedId;
    m_people = people;
    int total = 0;
    int selected = -1;
    {
        const QSignalBlocker blocker(m_table);
        m_table->setRowCount(people.size());
        for (int row = 0; row < people.size(); ++row) {
            const auto &person = people[row];
            const QStringList cells{person.profile.personId, person.profile.displayName,
                person.profile.group == "out" ? tr("组外") : tr("组内"),
                QString::number(person.sampleCount), person.enabled ? tr("启用") : tr("禁用")};
            for (int column = 0; column < cells.size(); ++column)
                m_table->setItem(row, column, new QTableWidgetItem(cells[column]));
            total += person.sampleCount;
            if (person.profile.personId == previousId) selected = row;
        }
        m_table->clearSelection();
    }
    m_summary->setText(tr("共%1人，%2张样本").arg(people.size()).arg(total));
    m_refresh->setEnabled(true);
    m_table->setEnabled(true);
    filterPeople();
    if (selected < 0 || m_table->isRowHidden(selected)) {
        selected = -1;
        for (int row = 0; row < people.size(); ++row)
            if (!m_table->isRowHidden(row)) { selected = row; break; }
    }
    if (selected >= 0) m_table->selectRow(selected);
    else selectPerson();
    m_message->setText(people.isEmpty() ? tr("人脸库为空，请在主界面采集或导入照片。") : tr("选择人员可编辑资料并查看样本。"));
}

void FaceDatabaseDialog::selectPerson() {
    const auto rows = m_table->selectionModel()->selectedRows();
    const int row = rows.isEmpty() ? -1 : rows.first().row();
    m_selectedId.clear();
    m_samples->clear();
    if (row < 0 || row >= m_people.size() || m_table->isRowHidden(row)) {
        m_id->clear(); m_name->clear(); m_enabled->setChecked(false);
        setEditingEnabled(false);
        return;
    }
    const auto &person = m_people[row];
    m_selectedId = person.profile.personId;
    m_id->setText(m_selectedId);
    m_name->setText(person.profile.displayName);
    m_group->setCurrentIndex(m_group->findData(person.profile.group));
    m_enabled->setChecked(person.enabled);
    setEditingEnabled(true);
    emit samplesRequested(m_selectedId);
}

void FaceDatabaseDialog::filterPeople() {
    const auto query = m_search->text().trimmed();
    for (int row = 0; row < m_people.size(); ++row) {
        const auto &profile = m_people[row].profile;
        m_table->setRowHidden(row, !profile.personId.contains(query, Qt::CaseInsensitive)
                                  && !profile.displayName.contains(query, Qt::CaseInsensitive));
    }
    const auto rows = m_table->selectionModel()->selectedRows();
    if (!rows.isEmpty() && m_table->isRowHidden(rows.first().row())) m_table->clearSelection();
}

void FaceDatabaseDialog::setSamples(const QString &personId, const FaceDatabaseSamples &samples) {
    if (personId != m_selectedId) return;
    m_samples->clear();
    int index = 0;
    for (const auto &sample : samples) {
        ++index;
        auto *item = new QListWidgetItem(sample.image.isNull() ? tr("照片不可用") : tr("样本%1").arg(index), m_samples);
        if (!sample.image.isNull()) item->setIcon(QPixmap::fromImage(sample.image));
        item->setData(Qt::UserRole, sample.sampleId);
        item->setToolTip(sample.details);
    }
}
