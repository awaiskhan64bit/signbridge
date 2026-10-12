import QtQuick
import SignBridge
import QtQuick.Controls

Switch {
    id: root

    property string label: ""

    indicator: Rectangle {
        implicitWidth: 42
        implicitHeight: 24
        radius: 12
        color: root.checked ? Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.28)
                             : Theme.surface3
        border.width: 1
        border.color: root.checked ? Theme.accent : Theme.stroke

        Behavior on color { ColorAnimation { duration: Theme.fast } }
        Behavior on border.color { ColorAnimation { duration: Theme.fast } }

        Rectangle {
            width: 18
            height: 18
            radius: 9
            y: 3
            x: root.checked ? parent.width - width - 3 : 3
            color: root.checked ? Theme.accent : Theme.textFaint

            Behavior on x { NumberAnimation { duration: Theme.normal; easing.type: Easing.OutCubic } }
            Behavior on color { ColorAnimation { duration: Theme.fast } }
        }
    }

    contentItem: Text {
        leftPadding: root.indicator.width + 12
        text: root.label
        color: Theme.textDim
        font.family: Theme.sans
        font.pixelSize: Theme.fontBody
        verticalAlignment: Text.AlignVCenter
    }
}
