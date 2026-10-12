import QtQuick
import SignBridge

/*
 * One sign in the phrase being built. Pops in when it lands, which is the only
 * motion in the app that celebrates, and it is on the thing the user produced.
 */
Rectangle {
    id: root

    property string text: ""
    property bool latest: false

    readonly property color tone: latest ? Theme.accent : Theme.violet

    implicitWidth: label.implicitWidth + 26
    implicitHeight: 34
    radius: height / 2
    color: Qt.rgba(tone.r, tone.g, tone.b, latest ? 0.22 : 0.12)
    border.width: 1
    border.color: Qt.rgba(tone.r, tone.g, tone.b, latest ? 0.65 : 0.30)

    scale: 0.85
    opacity: 0.0

    Component.onCompleted: {
        opacity = 1
        scale = 1
    }

    Behavior on opacity { NumberAnimation { duration: Theme.normal; easing.type: Easing.OutCubic } }
    Behavior on scale { NumberAnimation { duration: Theme.normal; easing.type: Easing.OutBack } }

    SequentialAnimation {
        running: root.latest
        loops: Animation.Infinite
        alwaysRunToEnd: true
        NumberAnimation {
            target: root; property: "scale"
            to: 1.04; duration: 900; easing.type: Easing.InOutSine
        }
        NumberAnimation {
            target: root; property: "scale"
            to: 1.0;  duration: 900; easing.type: Easing.InOutSine
        }
    }

    Text {
        id: label
        anchors.centerIn: parent
        text: root.text
        color: Theme.text
        font.family: Theme.sans
        font.pixelSize: Theme.fontBody
        font.weight: Font.DemiBold
    }
}
