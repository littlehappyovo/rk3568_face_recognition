#pragma once

#include "backendtypes.h"
#include <QDialog>
#include <functional>

class FaceBackendService;
class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QTableWidget;

class FaceDatabaseDialog final : public QDialog {
    Q_OBJECT
public:
    explicit FaceDatabaseDialog(FaceBackendService *backend, QWidget *parent = nullptr);
    void setAuthorization(std::function<void(std::function<void()>)> authorize) { m_authorize=std::move(authorize); }
signals:
    void refreshRequested();
    void samplesRequested(const QString &personId);
    void updatePersonRequested(const PersonProfile &person, bool enabled);
    void deletePersonRequested(const QString &personId);
    void deleteSampleRequested(const QString &personId, const QString &sampleId);
    void capturePersonSelected(const PersonProfile &person);
protected:
    void showEvent(QShowEvent *event) override;
private:
    void setPeople(const FaceDatabasePeople &people);
    void selectPerson();
    void filterPeople();
    void setSamples(const QString &personId, const FaceDatabaseSamples &samples);
    void setEditingEnabled(bool enabled);
    void setPending();
    std::function<void(std::function<void()>)> m_authorize;
    FaceDatabasePeople m_people;
    QString m_selectedId;
    QLineEdit *m_search = nullptr;
    QTableWidget *m_table = nullptr;
    QLineEdit *m_id = nullptr;
    QLineEdit *m_name = nullptr;
    QComboBox *m_group = nullptr;
    QCheckBox *m_enabled = nullptr;
    QListWidget *m_samples = nullptr;
    QLabel *m_summary = nullptr;
    QLabel *m_message = nullptr;
    QPushButton *m_refresh = nullptr;
    QPushButton *m_save = nullptr;
    QPushButton *m_deletePerson = nullptr;
    QPushButton *m_deleteSample = nullptr;
    QPushButton *m_capture = nullptr;
};
