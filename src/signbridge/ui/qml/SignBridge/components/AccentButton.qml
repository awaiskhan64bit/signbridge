import QtQuick
import QtQuick.Controls
import SignBridge

Button {
    id: root

    property bool primary: false
    property bool danger: false

    implicitHeight: 40
    implicitWidth: Math.max(96, label.implicitWidth + 34)
    hoverEnabled: true

    readonly property color tone: danger ? Theme.bad : Theme.accent

    background: Rectangle {
        radius: Theme.radiusSmall
        color: root.primary
               ? Qt.rgba(root.tone.r, root.tone.g, root.tone.b, root.down ? 0.28 : 0.16)
               : Qt.rgba(1, 1, 1, root.down ? 0.10 : (root.hovered ? 0.06 : 0.03))
        border.width: 1
        border.color: root.primary
                      ? Qt.rgba(root.tone.r, root.tone.g, root.tone.b, 0.55)
                      : Theme.stroke

        Behavior on color { ColorAnimation { duration: Theme.fast } }
        Behavior on border.color { ColorAnimation { duration: Theme.fast } }
    }

    contentItem: Text {
        id: label
        text: root.text
        color: root.primary ? root.tone : (root.hovered ? Theme.text : Theme.textDim)
        font.family: Theme.sans
        font.pixelSize: Theme.fontBody
        font.weight: Font.DemiBold
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    scale: root.down ? 0.97 : 1.0
    Behavior on scale { NumberAnimation { duration: Theme.fast; easing.type: Easing.OutCubic } }
}
