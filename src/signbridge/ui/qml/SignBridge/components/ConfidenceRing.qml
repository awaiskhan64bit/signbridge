import QtQuick
import SignBridge
import QtQuick.Shapes

/*
 * Confidence dial. The arc sweeps from 12 o'clock clockwise and the colour moves
 * from red through amber to green, so a glance at the colour is enough.
 */
Item {
    id: root

    property real value: 0.0          // 0..1
    property real thickness: 9
    property string caption: ""
    property string subcaption: ""
    property bool showValue: true

    implicitWidth: 132
    implicitHeight: 132

    readonly property real sweep: Math.max(0.001, Math.min(1, value) * 360)
    readonly property color tone: Theme.accentFor(value)

    Shape {
        anchors.fill: parent
        preferredRendererType: Shape.CurveRenderer
        layer.enabled: true
        layer.samples: 4

        ShapePath {
            strokeColor: Theme.surface3
            strokeWidth: root.thickness
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap
            PathAngleArc {
                centerX: root.width / 2
                centerY: root.height / 2
                radiusX: root.width / 2 - root.thickness
                radiusY: root.height / 2 - root.thickness
                startAngle: -90
                sweepAngle: 360
            }
        }

        ShapePath {
            strokeColor: root.tone
            strokeWidth: root.thickness
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap

            PathAngleArc {
                centerX: root.width / 2
                centerY: root.height / 2
                radiusX: root.width / 2 - root.thickness
                radiusY: root.height / 2 - root.thickness
                startAngle: -90
                sweepAngle: root.sweep

                // Tween the sweep itself so a jumpy model still glides.
                PropertyAnimation on sweepAngle {
                    duration: Theme.fast
                    easing.type: Easing.OutCubic
                }
            }
        }
    }

    Column {
        anchors.centerIn: parent
        spacing: 1

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            visible: root.showValue
            text: Math.round(root.value * 100) + "%"
            color: Theme.text
            font.family: Theme.mono
            font.pixelSize: 30
            font.weight: Font.Bold
        }

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            visible: root.caption !== ""
            text: root.caption
            color: Theme.textFaint
            font.family: Theme.sans
            font.pixelSize: Theme.fontMicro
            font.capitalization: Font.AllUppercase
            font.letterSpacing: 0.8
        }
    }
}
