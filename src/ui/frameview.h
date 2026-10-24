#pragma once

#include "backendtypes.h"

#include <QWidget>

class QPainter;

class FrameView final : public QWidget
{
    Q_OBJECT

public:
    explicit FrameView(QWidget *parent = nullptr);

    QSize sizeHint() const override;
    void setPlaceholder(const QString &text);
    void setFrame(const QImage &image,
                  const QVector<FaceObservation> &faces = {},
                  bool showOverlay = false);
    void setReferenceCards(const QVector<ReferenceFaceCard> &cards);
    void clearFrame();

signals:
    void faceClicked(qint64 trackId);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;

private:
    QRectF displayedImageRect() const;
    QColor colorFor(const FaceObservation &face) const;
    QString labelFor(const FaceObservation &face) const;
    void paintReferenceCards(QPainter &painter) const;

    QImage m_image;
    QVector<FaceObservation> m_faces;
    QVector<ReferenceFaceCard> m_referenceCards;
    QString m_placeholder;
    bool m_showOverlay = false;
};
