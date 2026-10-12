import QtQuick
import SignBridge

Rectangle {
    id: root

    property string label: ""
    property real value: 0.0
    property color tone: Theme.accent
    property bool showValue: true

    implicitHeight: label ? 34 : 12
    radius: 4
    color: "transparent"

    Text {
        id: text
        visible: root.label !== ""
        text: root.label
        color: Theme.textDim
        font.family: Theme.sans
        font.pixelSize: Theme.fontBody
        width: parent.width - (root.showValue ? 52 : 0)
        elide: Text.ElideRight
        anchors.top: parent.top
    }

    Text {
        visible: root.label !== "" && root.showValue
        text: Math.round(root.value * 100) + "%"
        color: root.tone
        font.family: Theme.mono
        font.pixelSize: Theme.fontSmall
        anchors.right: parent.right
        anchors.top: parent.top
    }

    Rectangle {
        id: track
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: text.visible ? text.bottom : parent.top
        anchors.topMargin: 6
        height: 5
        radius: 2.5
        color: Theme.surface3

        Rectangle {
            height: parent.height
            radius: parent.radius
            color: root.tone
            width: parent.width * Math.max(0, Math.min(1, root.value))

            Behavior on width {
                NumberAnimation { duration: Theme.fast; easing.type: Easing.OutCubic }
            }
        }
    }
}
