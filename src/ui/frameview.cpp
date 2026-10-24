#include "frameview.h"

#include <QFontMetrics>
#include <QMouseEvent>
#include <QPainter>
#include <QPaintEvent>
#include <QPen>

// FrameView 是一个被动绘图控件：backend 给它 QImage 和结构化的人脸结果，
// 它在 paintEvent() 中统一绘制。这样 OpenCV 不需要直接画 Qt 窗口。
FrameView::FrameView(QWidget *parent)
    : QWidget(parent)
    , m_placeholder(tr("等待视频输入"))
{
    setMinimumSize(360, 240);
    setAutoFillBackground(false);
}

QSize FrameView::sizeHint() const
{
    return {640, 400};
}

void FrameView::setPlaceholder(const QString &text)
{
    m_placeholder = text;
    update();
}

void FrameView::setFrame(const QImage &image,
                         const QVector<FaceObservation> &faces,
                         bool showOverlay)
{
    // FrameResult 可能来自后台线程；copy() 让控件拥有自己的像素副本，避免
    // 后台下一帧复用/释放缓冲区时，主线程正在绘制同一块内存。
    m_image = image.copy();
    m_faces = faces;
    m_referenceCards.clear();
    m_showOverlay = showOverlay;
    update();
}

void FrameView::setReferenceCards(const QVector<ReferenceFaceCard> &cards)
{
    m_image = {};
    m_faces.clear();
    m_referenceCards = cards;
    m_showOverlay = false;
    update();
}

void FrameView::clearFrame()
{
    m_image = {};
    m_faces.clear();
    m_referenceCards.clear();
    update();
}

QRectF FrameView::displayedImageRect() const
{
    if (m_image.isNull()) {
        return {};
    }
    QSizeF displaySize = m_image.size();
    displaySize.scale(size(), Qt::KeepAspectRatio);
    return {(width() - displaySize.width()) / 2.0,
            (height() - displaySize.height()) / 2.0,
            displaySize.width(), displaySize.height()};
}

void FrameView::mousePressEvent(QMouseEvent *event)
{
    // 鼠标坐标属于控件，检测框坐标是相对图像的 0~1 归一化坐标。这里先把
    // 点击位置换算回归一化图像坐标，再选择包含它的最小人脸框，支持多人重叠。
    const QRectF imageRect = displayedImageRect();
    if (!m_showOverlay || imageRect.isEmpty() || !imageRect.contains(event->pos())) {
        QWidget::mousePressEvent(event);
        return;
    }

    const QPointF normalized((event->pos().x() - imageRect.left()) / imageRect.width(),
                             (event->pos().y() - imageRect.top()) / imageRect.height());
    const FaceObservation *best = nullptr;
    qreal bestArea = 0.0;
    for (const FaceObservation &face : m_faces) {
        const QRectF box = face.normalizedBox.normalized();
        if (!box.contains(normalized)) {
            continue;
        }
        const qreal area = box.width() * box.height();
        if (!best || area < bestArea) {
            best = &face;
            bestArea = area;
        }
    }
    if (best) {
        emit faceClicked(best->trackId);
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

QColor FrameView::colorFor(const FaceObservation &face) const
{
    if (face.color.isValid()) {
        return face.color;
    }

    static const QColor palette[] = {
        QColor("#22c55e"), QColor("#38bdf8"), QColor("#f59e0b"),
        QColor("#e879f9"), QColor("#fb7185"), QColor("#a3e635")
    };
    const int index = qAbs(face.trackId) % (sizeof(palette) / sizeof(palette[0]));
    return palette[index];
}

QString FrameView::labelFor(const FaceObservation &face) const
{
    QString label;
    switch (face.identityState) {
    case IdentityState::Known:
        label = face.displayName.isEmpty() ? face.personId : face.displayName;
        if (!face.personId.isEmpty() && label != face.personId) {
            label += QStringLiteral(" [") + face.personId + QStringLiteral("]");
        }
        if (!face.group.isEmpty()) {
            label += QStringLiteral(" · ") + face.group;
        }
        break;
    case IdentityState::Unknown:
        label = tr("未知人员");
        break;
    case IdentityState::DetectionOnly:
        label = tr("目标 #%1").arg(face.trackId);
        break;
    }

    if (!face.scoreText.isEmpty()) {
        // Keep the complete identity and score on one line. The label box is
        // allowed to grow instead of wrapping or eliding the text.
        label += QStringLiteral("  ") + face.scoreText;
    }
    if (face.selectedForCapture) {
        label.prepend(tr("采集目标 · "));
    }
    return label;
}

void FrameView::paintEvent(QPaintEvent *event)
{
    // Qt 要求 QWidget 的绘制发生在主线程。m_image/m_faces 是上一次 setFrame()
    // 保存的快照，因此 paintEvent() 不需要接触摄像头线程或 cv::Mat。
    Q_UNUSED(event)

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(rect(), QColor("#111827"));

    if (!m_referenceCards.isEmpty()) {
        paintReferenceCards(painter);
        return;
    }

    if (m_image.isNull()) {
        painter.setPen(QColor("#94a3b8"));
        painter.drawText(rect(), Qt::AlignCenter, m_placeholder);
        return;
    }

    const QRectF imageRect = displayedImageRect();
    painter.drawImage(imageRect, m_image);

    if (!m_showOverlay) {
        return;
    }

    for (const FaceObservation &face : m_faces) {
        // normalizedBox 是相对图像的 0~1 坐标；先映射到实际显示区域，再裁剪到
        // imageRect，才能兼容控件比例变化和 KeepAspectRatio 留下的黑边。
        const QRectF normalized = face.normalizedBox.normalized();
        QRectF box(imageRect.left() + normalized.left() * imageRect.width(),
                   imageRect.top() + normalized.top() * imageRect.height(),
                   normalized.width() * imageRect.width(),
                   normalized.height() * imageRect.height());
        box = box.intersected(imageRect);
        if (box.isEmpty()) {
            continue;
        }

        const QColor color = colorFor(face);
        painter.setBrush(Qt::NoBrush);
        if (face.selectedForCapture) {
            painter.setPen(QPen(QColor("#ffffff"), 7.0));
            painter.drawRoundedRect(box, 5.0, 5.0);
        }
        painter.setPen(QPen(color, face.selectedForCapture ? 4.0 : 3.0));
        painter.drawRoundedRect(box, 4.0, 4.0);

        const QString label = labelFor(face);
        const QFontMetrics metrics(painter.font());
        int contentWidth = 0;
#if QT_VERSION >= QT_VERSION_CHECK(5, 11, 0)
        contentWidth = metrics.horizontalAdvance(label);
#else
        contentWidth = metrics.width(label);
#endif
        const int labelWidth = qMax(160, contentWidth + 12);
        const int textHeight = metrics.height() + 8;

        qreal labelLeft = box.left();
        if (labelLeft + labelWidth > imageRect.right()) {
            labelLeft = imageRect.right() - labelWidth;
        }
        labelLeft = qMax(imageRect.left(), labelLeft);

        qreal labelTop = box.top() - textHeight;
        if (labelTop < imageRect.top()) {
            labelTop = box.bottom();
        }
        if (labelTop + textHeight > imageRect.bottom()) {
            labelTop = imageRect.bottom() - textHeight;
        }
        labelTop = qMax(imageRect.top(), labelTop);

        const QRectF labelRect(labelLeft, labelTop, labelWidth, textHeight);

        QColor background = color;
        background.setAlpha(220);
        painter.setPen(Qt::NoPen);
        painter.setBrush(background);
        painter.drawRoundedRect(labelRect, 3.0, 3.0);
        painter.setPen(QColor("#07111f"));
        painter.drawText(labelRect.adjusted(6, 2, -6, -2),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         label);
    }
}

void FrameView::paintReferenceCards(QPainter &painter) const
{
    const int margin = 12;
    const int gap = 10;
    const int columns = width() >= 520 ? 2 : 1;
    const int cardWidth = qMax(120, (width() - margin * 2 - gap * (columns - 1)) / columns);
    const int cardHeight = 190;

    for (int index = 0; index < m_referenceCards.size(); ++index) {
        const ReferenceFaceCard &card = m_referenceCards.at(index);
        const int row = index / columns;
        const int column = index % columns;
        const QRect cardRect(margin + column * (cardWidth + gap),
                             margin + row * (cardHeight + gap),
                             cardWidth, cardHeight);
        painter.setPen(QPen(QColor("#334155"), 1));
        painter.setBrush(QColor("#1e293b"));
        painter.drawRoundedRect(cardRect, 8, 8);

        const QRect imageRect = cardRect.adjusted(8, 8, -8, -48);
        if (card.image.isNull()) {
            painter.setPen(QColor("#94a3b8"));
            painter.drawText(imageRect, Qt::AlignCenter, tr("无采集照片"));
        } else {
            QSize imageSize = card.image.size();
            imageSize.scale(imageRect.size(), Qt::KeepAspectRatio);
            const QRect fitted(imageRect.left() + (imageRect.width() - imageSize.width()) / 2,
                               imageRect.top() + (imageRect.height() - imageSize.height()) / 2,
                               imageSize.width(), imageSize.height());
            painter.drawImage(fitted, card.image);
        }

        QString title = card.displayName.isEmpty() ? card.personId : card.displayName;
        if (!card.personId.isEmpty() && title != card.personId) {
            title += QStringLiteral(" [") + card.personId + QStringLiteral("]");
        }
        if (!card.group.isEmpty()) {
            title += QStringLiteral(" · ") + card.group;
        }
        painter.setPen(QColor("#e2e8f0"));
        painter.drawText(cardRect.adjusted(10, cardHeight - 38, -10, -18),
                         Qt::AlignLeft | Qt::AlignVCenter,
                         title);
        painter.setPen(QColor("#86efac"));
        painter.drawText(cardRect.adjusted(10, cardHeight - 20, -10, -4),
                         Qt::AlignLeft | Qt::AlignVCenter,
                         tr("已确认"));
    }

    const int rows = (m_referenceCards.size() + columns - 1) / columns;
    if (rows * (cardHeight + gap) + margin > height()) {
        painter.setPen(QColor("#fbbf24"));
        painter.drawText(rect().adjusted(12, 0, -12, -8),
                         Qt::AlignRight | Qt::AlignBottom,
                         tr("共 %1 人").arg(m_referenceCards.size()));
    }
}
