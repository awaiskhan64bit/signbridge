import QtQuick
import SignBridge

/*
 * A panel with an optional header.
 *
 * Height has to work both ways: a card dropped into a ColumnLayout should size
 * itself to its content, and the same card given Layout.fillHeight should hand
 * the slack to its body so a list inside it can scroll. Binding the body to the
 * parent's height rather than to a fixed number gets both from one expression.
 */
Rectangle {
    id: root

    property string title: ""
    property string subtitle: ""
    property int headerHeight: title ? 58 : 0
    property int bodyGap: title ? Theme.gap : 0
    property alias headerContent: headerExtra.data
    default property alias content: body.data

    readonly property int bodyTop: Theme.pad + headerHeight + bodyGap

    color: Theme.surface2
    radius: Theme.radius
    border.width: 1
    border.color: Theme.stroke

    implicitHeight: bodyTop + body.implicitHeight + Theme.pad

    // ----------------------------------------------------------------- header

    Row {
        id: header
        anchors.top: parent.top
        anchors.topMargin: Theme.pad
        anchors.left: parent.left
        anchors.leftMargin: Theme.pad
        anchors.right: parent.right
        anchors.rightMargin: Theme.pad
        height: root.headerHeight
        visible: root.title !== ""
        spacing: Theme.gap

        Column {
            width: parent.width - headerExtra.width
            anchors.verticalCenter: parent.verticalCenter
            spacing: 3

            Text {
                width: parent.width
                text: root.title
                color: Theme.text
                font.family: Theme.sans
                font.pixelSize: Theme.fontTitle
                font.weight: Font.DemiBold
                elide: Text.ElideRight
            }
            Text {
                width: parent.width
                visible: root.subtitle !== ""
                text: root.subtitle
                color: Theme.textFaint
                font.family: Theme.sans
                font.pixelSize: Theme.fontSmall
                elide: Text.ElideRight
            }
        }

        Row {
            id: headerExtra
            anchors.verticalCenter: parent.verticalCenter
            spacing: 6
        }
    }

    // ------------------------------------------------------------------- body

    Item {
        id: body
        anchors.top: header.visible ? header.bottom : parent.top
        anchors.topMargin: root.bodyGap + (header.visible ? 0 : Theme.pad)
        anchors.left: parent.left
        anchors.leftMargin: Theme.pad
        anchors.right: parent.right
        anchors.rightMargin: Theme.pad
        height: Math.max(0, root.height - root.bodyTop - Theme.pad)

        // Lets the card report a sensible implicit height to the layout.
        implicitHeight: childrenRect.height
    }
}
