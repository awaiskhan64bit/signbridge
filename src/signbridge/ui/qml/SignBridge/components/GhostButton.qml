import QtQuick
import QtQuick.Controls
import SignBridge

Button {
    id: root

    implicitHeight: 34
    hoverEnabled: true

    background: Rectangle {
        radius: Theme.radiusSmall
        color: root.hovered ? Qt.rgba(1, 1, 1, 0.07) : "transparent"
        border.width: 1
        border.color: root.hovered ? Theme.strokeBright : "transparent"
        Behavior on color { ColorAnimation { duration: Theme.fast } }
    }

    contentItem: Text {
        text: root.text
        color: root.hovered ? Theme.text : Theme.textDim
        font.family: Theme.sans
        font.pixelSize: Theme.fontSmall
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
