import QtQuick
import SignBridge
import QtQuick.Controls

Row {
    id: root

    property string label: ""
    property real from: 0
    property real to: 1
    property real stepSize: 0.05
    property real value: 1.0

    // Emitted only when the user actually moves the slider. A plain value
    // change also fires when the binding is first evaluated, and at that point
    // the value is not always meaningful. Emitting that back into the settings
    // once wrote a NaN into the persisted threshold, which silently disabled
    // every later confidence comparison.
    signal moved(real value)

    spacing: 14

    Text {
        id: caption
        width: 96
        anchors.verticalCenter: parent.verticalCenter
        text: root.label
        color: Theme.textDim
        font.family: Theme.sans
        font.pixelSize: Theme.fontBody
    }

    Slider {
        id: slider
        anchors.verticalCenter: parent.verticalCenter
        width: 150
        height: 24
        from: root.from
        to: root.to
        stepSize: root.stepSize
        // Bind one way. Writing root.value back from here would fight the
        // binding and, worse, persist a not-yet-meaningful value on startup.
        value: root.value
        onMoved: {
            root.value = value
            root.moved(value)
        }

        background: Rectangle {
            x: slider.leftPadding
            y: slider.topPadding + slider.availableHeight / 2 - height / 2
            width: slider.availableWidth
            height: 4
            radius: 2
            color: Theme.surface3

            Rectangle {
                width: slider.visualPosition * parent.width
                height: parent.height
                radius: parent.radius
                color: Theme.accent
            }
        }

        handle: Rectangle {
            x: slider.leftPadding + slider.visualPosition * (slider.availableWidth - width)
            y: slider.topPadding + slider.availableHeight / 2 - height / 2
            width: 14
            height: 14
            radius: 7
            color: Theme.text
            border.width: 2
            border.color: Theme.bg
        }
    }

    Text {
        anchors.verticalCenter: parent.verticalCenter
        width: 46
        text: root.value.toFixed(2) + "x"
        color: Theme.text
        font.family: Theme.mono
        font.pixelSize: Theme.fontSmall
    }
}
