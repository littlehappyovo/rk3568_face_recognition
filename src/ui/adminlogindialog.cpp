#include "adminlogindialog.h"
#include "accessclient.h"
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QShowEvent>
#include <QTimer>
#include <QFrame>
AdminLoginDialog::AdminLoginDialog(AccessClient *client, QWidget *parent)
    : QDialog(parent),m_client(client) {
    setWindowTitle(tr("管理员认证")); resize(760,590);
    if (parent) {
        // Rotated EGLFS secondary native windows can disagree with input geometry.
        // Keep the complete login surface in the main window's coordinate space.
        setParent(parent,Qt::Widget); setModal(false);
        setObjectName("adminLoginOverlay"); setAttribute(Qt::WA_StyledBackground);
        setStyleSheet("#adminLoginOverlay { background-color:rgba(15,23,42,180); }"
                      "#adminLoginCard { background-color:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; }");
        setGeometry(parent->rect()); parent->installEventFilter(this);
    } else setModal(true);
    auto *outer=new QVBoxLayout(this); outer->setContentsMargins(24,24,24,24);
    outer->addStretch();
    auto *card=new QFrame(this); card->setObjectName("adminLoginCard");
    card->setMinimumWidth(760); card->setMaximumWidth(820);
    auto *layout=new QVBoxLayout(card); layout->setContentsMargins(20,16,20,16);
    layout->addWidget(new QLabel(tr("管理员认证"),card));
    outer->addWidget(card,0,Qt::AlignHCenter); outer->addStretch();
    m_instruction=new QLabel(tr("正在连接管理服务…"),card); layout->addWidget(m_instruction);
    m_password=new QLineEdit(this); m_password->setObjectName("adminPassword");
    m_confirm=new QLineEdit(this); m_confirm->setObjectName("adminPasswordConfirm");
    m_confirmLabel=new QLabel(tr("再次输入密码"),this);
    layout->addWidget(new QLabel(tr("管理密码"),this)); layout->addWidget(m_password);
    layout->addWidget(m_confirmLabel); layout->addWidget(m_confirm);
    for(auto *edit:{m_password,m_confirm}) { edit->setEchoMode(QLineEdit::Password); edit->setMaxLength(128); }
    m_confirm->hide(); m_confirmLabel->hide();
    m_error=new QLabel(this); m_error->setObjectName("adminError"); m_error->setTextFormat(Qt::PlainText);
    m_error->setWordWrap(true); m_error->setStyleSheet("color:#b42318;"); layout->addWidget(m_error);
    auto *buttons=new QDialogButtonBox(QDialogButtonBox::Cancel,this);
    m_submit=buttons->addButton(tr("登录"),QDialogButtonBox::AcceptRole); m_submit->setObjectName("adminSubmit");
    connect(m_submit,&QPushButton::clicked,this,&AdminLoginDialog::submit);
    connect(buttons,&QDialogButtonBox::rejected,this,&QDialog::reject);
    connect(m_password,&QLineEdit::returnPressed,this,&AdminLoginDialog::submit);
    connect(m_confirm,&QLineEdit::returnPressed,this,&AdminLoginDialog::submit); layout->addWidget(buttons);
    pending(true);
    if(client) {
        connect(client,&AccessClient::authStatusReady,this,[this](bool initialized) {
            m_setup=!initialized; m_ready=true;
            m_confirm->setVisible(m_setup); m_confirmLabel->setVisible(m_setup);
            m_instruction->setText(m_setup ? tr("首次使用，请在板卡设置8至128字符的独立管理密码。") : tr("输入管理密码进入管理界面。"));
            m_submit->setText(m_setup ? tr("设置并进入") : tr("登录")); pending(false);
        });
        connect(client,&AccessClient::authenticated,this,[this](const QString &,const QDateTime &) {
            if(!m_submitted || !isVisible()) return;
            m_password->clear(); m_confirm->clear(); m_submitted=false; accept();
        });
        connect(client,&AccessClient::errorOccurred,this,[this](const QString &message) {
            if(!isVisible()) return;
            m_submitted=false; m_error->setText(message); pending(false);
        });
    }
    connect(this,&QDialog::rejected,this,[this] { if(m_client) m_client->logout(); m_password->clear(); m_confirm->clear(); });
}
void AdminLoginDialog::open() {
    if (!parentWidget()) { QDialog::open(); return; }
    setGeometry(parentWidget()->rect()); show(); raise(); m_password->setFocus();
}
void AdminLoginDialog::showEvent(QShowEvent *event) {
    QDialog::showEvent(event);
    QTimer::singleShot(0,this,[this] { if(m_client) m_client->authStatus(); else { m_error->setText(tr("管理服务尚未连接")); pending(false); } });
}
bool AdminLoginDialog::eventFilter(QObject *object,QEvent *event) {
    if(object==parentWidget() && event->type()==QEvent::Resize) setGeometry(parentWidget()->rect());
    return QDialog::eventFilter(object,event);
}
void AdminLoginDialog::pending(bool value) {
    m_submit->setEnabled(!value); m_password->setEnabled(!value); m_confirm->setEnabled(!value);
    if(!value && isVisible()) m_password->setFocus();
}
void AdminLoginDialog::submit() {
    if(!m_submit->isEnabled() || !m_client) return;
    if(!m_ready) { pending(true); m_client->authStatus(); return; }
    const auto password=m_password->text();
    if(password.size()<8 || password.size()>128) { m_error->setText(tr("密码长度须为8至128字符")); return; }
    if(m_setup && password!=m_confirm->text()) { m_error->setText(tr("两次输入的密码不一致")); return; }
    m_error->clear(); m_submitted=true; pending(true);
    if(m_setup) m_client->setupPassword(password); else m_client->login(password);
}
