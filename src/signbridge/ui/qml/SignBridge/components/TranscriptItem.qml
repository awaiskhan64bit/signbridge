import QtQuick
import SignBridge
import QtQuick.Layouts

Rectangle {
    id: root

    property string text: ""
    property string time: ""
    property bool spoken: true

    implicitHeight: layout.implicitHeight + 22
    radius: Theme.radiusSmall
    color: Theme.surface

    border.width: 1
    border.color: root.spoken ? Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.22)
                              : Theme.stroke

    RowLayout {
        id: layout
        anchors.fill: parent
        anchors.margins: 11
        spacing: 12

        Rectangle {
            width: 6
            height: 6
            radius: 3
            color: root.spoken ? Theme.good : Theme.textFaint
        }

        Text {
            Layout.fillWidth: true
            text: root.text
            color: Theme.text
            font.family: Theme.sans
            font.pixelSize: Theme.fontTitle
            wrapMode: Text.WordWrap
        }

        Text {
            text: root.time
            color: Theme.textFaint
            font.family: Theme.mono
            font.pixelSize: Theme.fontMicro
        }
    }
}
