import QtQuick
import SignBridge

Rectangle {
    id: root

    property string label: ""
    property string value: ""
    property color accentColor: Theme.text
    property bool live: false

    implicitWidth: row.implicitWidth + 22
    implicitHeight: 30
    radius: height / 2
    color: Qt.rgba(accentColor.r, accentColor.g, accentColor.b, 0.10)
    border.width: 1
    border.color: Qt.rgba(accentColor.r, accentColor.g, accentColor.b, 0.28)

    Row {
        id: row
        anchors.centerIn: parent
        spacing: 7

        Rectangle {
            width: 6; height: 6; radius: 3
            anchors.verticalCenter: parent.verticalCenter
            color: root.accentColor
            visible: root.live

            SequentialAnimation on opacity {
                running: root.live
                loops: Animation.Infinite
                NumberAnimation { to: 0.25; duration: 700; easing.type: Easing.InOutSine }
                NumberAnimation { to: 1.0;  duration: 700; easing.type: Easing.InOutSine }
            }
            onVisibleChanged: if (!visible) opacity = 1
        }

        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: root.label
            color: Theme.textFaint
            font.family: Theme.sans
            font.pixelSize: Theme.fontMicro
            font.capitalization: Font.AllUppercase
            font.letterSpacing: 0.6
        }

        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: root.value
            color: root.accentColor
            font.family: Theme.mono
            font.pixelSize: Theme.fontSmall
            font.weight: Font.DemiBold
        }
    }
}
