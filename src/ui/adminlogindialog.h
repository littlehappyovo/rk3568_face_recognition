#pragma once
#include <QDialog>
#include <QPointer>
class AccessClient;
class QLineEdit;
class QLabel;
class QPushButton;
class AdminLoginDialog final : public QDialog {
    Q_OBJECT
public:
    explicit AdminLoginDialog(AccessClient *client, QWidget *parent=nullptr);
    void open();
protected:
    bool eventFilter(QObject *object, QEvent *event) override;
    void showEvent(QShowEvent *event) override;
private:
    void submit();
    void pending(bool value);
    QPointer<AccessClient> m_client;
    QLineEdit *m_password, *m_confirm;
    QLabel *m_confirmLabel, *m_error, *m_instruction;
    QPushButton *m_submit;
    bool m_setup=false, m_ready=false, m_submitted=false;
};
