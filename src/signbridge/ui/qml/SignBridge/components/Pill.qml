import QtQuick
import SignBridge

Rectangle {
    id: root

    property string text: ""
    property color tone: Theme.accent
    property bool filled: false

    implicitWidth: label.implicitWidth + 20
    implicitHeight: 26
    radius: height / 2
    color: filled
           ? Qt.rgba(tone.r, tone.g, tone.b, 0.18)
           : Qt.rgba(tone.r, tone.g, tone.b, 0.07)
    border.width: 1
    border.color: Qt.rgba(tone.r, tone.g, tone.b, filled ? 0.55 : 0.25)

    Text {
        id: label
        anchors.centerIn: parent
        text: root.text
        color: root.filled ? root.tone : Theme.textDim
        font.family: Theme.sans
        font.pixelSize: Theme.fontBody
        font.weight: root.filled ? Font.DemiBold : Font.Normal
    }
}
