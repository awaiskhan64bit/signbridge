import QtQuick
import SignBridge
import QtQuick.Controls

/*
 * Segmented control. Kept in-house rather than using the Controls version so the
 * selection can animate as a sliding pill instead of snapping.
 */
Rectangle {
    id: root

    property var options: []       // [{ value, label }]
    property string current: ""

    signal picked(string value)

    implicitHeight: 36
    radius: Theme.radiusSmall
    color: Theme.surface3
    border.width: 1
    border.color: Theme.stroke
    clip: true

    Rectangle {
        id: thumb
        radius: Theme.radiusSmall - 2
        color: Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.16)
        border.width: 1
        border.color: Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.45)
        width: root.options.length > 0 ? root.width / root.options.length : 0
        height: parent.height - 4
        y: 2
        x: {
            var index = -1
            for (var i = 0; i < root.options.length; ++i) {
                if (root.options[i].value === root.current) { index = i; break }
            }
            return index < 0 ? 0 : index * width
        }

        Behavior on x {
            NumberAnimation { duration: Theme.normal; easing.type: Easing.OutCubic }
        }
    }

    Row {
        anchors.fill: parent
        Repeater {
            model: root.options

            delegate: Item {
                required property var modelData
                required property int index

                width: root.options.length > 0 ? root.width / root.options.length : 0
                height: root.height

                Text {
                    anchors.centerIn: parent
                    text: modelData.label
                    color: root.current === modelData.value ? Theme.text : Theme.textFaint
                    font.family: Theme.sans
                    font.pixelSize: Theme.fontSmall
                    font.weight: root.current === modelData.value ? Font.DemiBold : Font.Normal

                    Behavior on color { ColorAnimation { duration: Theme.fast } }
                }

                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: root.picked(modelData.value)
                }
            }
        }
    }
}
