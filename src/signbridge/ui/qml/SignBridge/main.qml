import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick.Shapes
import SignBridge

ApplicationWindow {
    id: window

    width: 1440
    height: 900
    minimumWidth: 1120
    minimumHeight: 720
    visible: true
    title: "SignBridge"
    color: Theme.bg

    property int railWidth: 400

    // ------------------------------------------------------------- background

    Rectangle {
        anchors.fill: parent

        gradient: Gradient {
            GradientStop { position: 0.0; color: Theme.bg }
            GradientStop { position: 0.55; color: Theme.bgLift }
            GradientStop { position: 1.0; color: Theme.bg }
        }
    }

    // A faint accent wash across the top, so the dark background has some depth
    // without competing with the camera image for attention.
    Rectangle {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 320
        opacity: 0.5

        gradient: Gradient {
            orientation: Gradient.Vertical
            GradientStop { position: 0.0; color: Qt.rgba(0.35, 0.88, 1.0, 0.08) }
            GradientStop { position: 1.0; color: "transparent" }
        }
    }

    // ----------------------------------------------------------------- header

    header: Rectangle {
        height: 72
        color: "transparent"

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: Theme.pad
            anchors.rightMargin: Theme.pad
            spacing: Theme.gap

            // Wordmark.
            RowLayout {
                spacing: 12

                Rectangle {
                    width: 38
                    height: 38
                    radius: 11
                    gradient: Gradient {
                        GradientStop { position: 0.0; color: Theme.accent }
                        GradientStop { position: 1.0; color: Theme.violet }
                    }

                    // A stylised signing hand, reduced to two arcs and a palm.
                    Canvas {
                        anchors.fill: parent
                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.reset()
                            ctx.strokeStyle = Theme.bg
                            ctx.lineWidth = 2.6
                            ctx.lineCap = "round"
                            ctx.beginPath()
                            ctx.arc(19, 21, 7.5, Math.PI * 0.75, Math.PI * 2.35)
                            ctx.stroke()
                            ctx.beginPath()
                            ctx.moveTo(13.5, 14.5)
                            ctx.lineTo(13.5, 9.5)
                            ctx.moveTo(18, 13.5)
                            ctx.lineTo(18, 8)
                            ctx.moveTo(22.5, 14)
                            ctx.lineTo(22.5, 9.5)
                            ctx.stroke()
                        }
                    }
                }

                ColumnLayout {
                    spacing: 0
                    Text {
                        text: "SignBridge"
                        color: Theme.text
                        font.family: Theme.sans
                        font.pixelSize: 21
                        font.weight: Font.Bold
                    }
                    Text {
                        text: "offline sign language to speech"
                        color: Theme.textFaint
                        font.family: Theme.sans
                        font.pixelSize: Theme.fontMicro
                    }
                }
            }

            Item { Layout.fillWidth: true }

            StatChip {
                label: "Engine"
                value: app.recognizerKind === "neural" ? "ML" : "Rules"
                accentColor: app.recognizerKind === "neural" ? Theme.good : Theme.warn
            }
            StatChip {
                label: "Voice"
                value: app.ttsEngine === "off" ? "Muted" : "Ready"
                accentColor: app.ttsEngine === "off" ? Theme.textFaint : Theme.violet
                live: app.ttsEngine !== "off"
            }
            StatChip {
                label: "Net"
                value: "Offline"
                accentColor: Theme.good
            }

            AccentButton {
                text: app.running ? "Stop" : "Start"
                primary: !app.running
                danger: app.running
                onClicked: app.toggleRunning()
            }
        }
    }

    // ------------------------------------------------------------------ body
    //
    // The two panes are positioned with anchors rather than laid out by a
    // RowLayout, because a layout would have to infer the camera pane's width
    // from its children and happily give the whole window to the rail.

    Item {
        id: body
        anchors.fill: parent
        anchors.leftMargin: Theme.pad
        anchors.rightMargin: Theme.pad
        anchors.bottomMargin: Theme.gap

        // ---------------------------------------------------------- camera

        Item {
            id: left
            anchors.left: parent.left
            anchors.right: rail.left
            anchors.rightMargin: Theme.gap
            anchors.top: parent.top
            anchors.bottom: parent.bottom

            readonly property real liveBarHeight: 132
            // The viewport takes whatever height is left over. The image inside
            // keeps its aspect and letterboxes against a panel background of the
            // same colour, so the bars are invisible rather than dead space.
            readonly property real cameraHeight: Math.max(
                200, height - liveBarHeight - Theme.gap
            )

            // The viewport. A tracking hand lights the border, so peripheral
            // vision can tell at a glance whether the camera sees anything.
            Rectangle {
                id: viewport
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                height: left.cameraHeight
                radius: Theme.radius
                color: Theme.bg
                border.width: 2
                border.color: app.tracking
                             ? Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.55)
                             : Theme.stroke
                clip: true

                Behavior on border.color { ColorAnimation { duration: Theme.normal } }

                // Vignette so the panel reads as a screen, not a hole.
                Rectangle {
                    anchors.fill: parent
                    gradient: Gradient {
                        orientation: Gradient.Vertical
                        GradientStop { position: 0.0; color: Qt.rgba(0, 0, 0, 0.35) }
                        GradientStop { position: 0.25; color: "transparent" }
                        GradientStop { position: 0.8; color: "transparent" }
                        GradientStop { position: 1.0; color: Qt.rgba(0, 0, 0, 0.45) }
                    }
                }

                Image {
                    id: camera
                    anchors.fill: parent
                    anchors.margins: 1
                    source: "image://frames/" + app.frameId
                    asynchronous: false
                    cache: false
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                    mipmap: false
                }

                // Placeholder for the moments before the first frame arrives.
                Column {
                    anchors.centerIn: parent
                    spacing: 10
                    visible: app.frameId === 0

                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: app.sourceLabel
                        color: Theme.textFaint
                        font.family: Theme.sans
                        font.pixelSize: Theme.fontBody
                    }
                }

                // Top left: live stats.
                Row {
                    anchors.left: parent.left
                    anchors.top: parent.top
                    anchors.margins: 14
                    spacing: 8

                    StatChip {
                        label: "FPS"
                        value: app.fps.toFixed(0)
                        accentColor: app.fps >= 24 ? Theme.good : Theme.warn
                    }
                    StatChip {
                        label: "Latency"
                        value: app.latency.toFixed(1) + "ms"
                        accentColor: app.latency < 40 ? Theme.good : Theme.warn
                    }
                    StatChip {
                        label: "Hands"
                        value: app.handCount
                        accentColor: app.tracking ? Theme.accent : Theme.textFaint
                    }
                }

                // Top right: source picker.
                Row {
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: 14
                    spacing: 8

                    Segmented {
                        width: 264
                        current: app.sourceLabel.indexOf("Simulated") >= 0 ? "simulated"
                                  : app.sourceLabel.indexOf("Clip") >= 0 ? "video" : "camera"
                        options: [
                            { value: "camera",    label: "Camera" },
                            { value: "video",     label: "Clip" },
                            { value: "simulated", label: "Demo" }
                        ]
                        onPicked: (value) => app.setSource(value)
                    }

                    GhostButton {
                        text: "Open..."
                        onClicked: clipDialog.open()
                    }
                }

                // Bottom: the window fill meter and any error.
                Column {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    anchors.margins: 14
                    spacing: 8

                    Rectangle {
                        anchors.left: parent.left
                        width: 190
                        height: 4
                        radius: 2
                        visible: app.running
                        color: Qt.rgba(1, 1, 1, 0.12)

                        Rectangle {
                            height: parent.height
                            radius: parent.radius
                            width: parent.width * app.windowFill
                            color: Theme.accent
                            Behavior on width { NumberAnimation { duration: Theme.normal } }
                        }
                    }

                    Rectangle {
                        anchors.left: parent.left
                        width: parent.width
                        height: visible ? 30 : 0
                        visible: app.errorText !== ""
                        radius: 8
                        color: Qt.rgba(Theme.bad.r, Theme.bad.g, Theme.bad.b, 0.14)

                        Text {
                            anchors.fill: parent
                            anchors.leftMargin: 12
                            anchors.rightMargin: 12
                            text: app.errorText
                            color: Theme.bad
                            font.family: Theme.sans
                            font.pixelSize: Theme.fontSmall
                            elide: Text.ElideRight
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
            }

            // ------------------------------------------------------ live sign

            Rectangle {
                id: liveBar
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                height: left.liveBarHeight
                radius: Theme.radius
                color: Theme.surface2
                border.width: 1
                border.color: Theme.stroke

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: Theme.pad
                    spacing: Theme.gap * 1.5

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 4

                        Text {
                            text: "CURRENTLY SIGNING"
                            color: Theme.textFaint
                            font.family: Theme.sans
                            font.pixelSize: Theme.fontMicro
                            font.letterSpacing: 1.2
                        }

                        Text {
                            Layout.fillWidth: true
                            text: app.pendingLabel !== "" ? app.pendingLabel : "..."
                            color: app.pendingLabel !== "" ? Theme.text : Theme.textFaint
                            font.family: Theme.sans
                            font.pixelSize: Theme.fontDisplay
                            font.weight: Font.Bold
                            elide: Text.ElideRight
                        }

                        Text {
                            visible: app.pendingSpeech !== ""
                            text: "speaks as \"" + app.pendingSpeech + "\""
                            color: Theme.accent
                            font.family: Theme.sans
                            font.pixelSize: Theme.fontSmall
                        }

                        Text {
                            Layout.fillWidth: true
                            visible: app.alternatives.length > 0
                            text: app.alternatives.map(a => a.label + " " + Math.round(a.score * 100) + "%").join("     ")
                            color: Theme.textFaint
                            font.family: Theme.mono
                            font.pixelSize: Theme.fontMicro
                            elide: Text.ElideRight
                        }
                    }

                    ConfidenceRing {
                        value: app.confidence
                        caption: "confidence"
                    }
                }
            }
        }

        // ---------------------------------------------------------------- rail

        ColumnLayout {
            id: rail
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: window.railWidth
            spacing: Theme.gap

            // Phrase being assembled.
            Card {
                Layout.fillWidth: true
                title: "Phrase"
                subtitle: app.phraseSigns.length + " sign" + (app.phraseSigns.length === 1 ? "" : "s")
                headerContent: [
                    GhostButton {
                        text: "Speak"
                        onClicked: app.flushPhrase()
                    },
                    GhostButton {
                        text: "Clear"
                        onClicked: app.clearPhrase()
                    }
                ]

                Flow {
                    width: parent.width
                    spacing: 8
                    visible: app.phraseSigns.length > 0

                    Repeater {
                        model: app.phraseSigns
                        delegate: SignChip {
                            required property var modelData
                            required property int index
                            text: modelData.text
                            latest: index === app.phraseSigns.length - 1
                        }
                    }
                }

                Text {
                    visible: app.phraseSigns.length === 0
                    width: parent.width
                    text: "Signs collect here, then are spoken when you pause."
                    color: Theme.textFaint
                    font.family: Theme.sans
                    font.pixelSize: Theme.fontSmall
                    wrapMode: Text.WordWrap
                }
            }

            // Spoken history.
            Card {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "Spoken"
                headerContent: [
                    GhostButton {
                        text: "Repeat"
                        onClicked: app.repeatLast()
                    },
                    GhostButton {
                        text: "Clear"
                        onClicked: app.clearHistory()
                    }
                ]

                ListView {
                    width: parent.width
                    height: parent.height
                    clip: true
                    spacing: 8
                    model: app.history
                    boundsBehavior: Flickable.StopAtBounds

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded
                    }

                    delegate: TranscriptItem {
                        required property var modelData
                        width: ListView.view.width
                        text: modelData.text
                        time: modelData.time
                        spoken: modelData.spoken
                    }

                    Text {
                        anchors.centerIn: parent
                        visible: app.history.length === 0
                        text: "Nothing spoken yet."
                        color: Theme.textFaint
                        font.family: Theme.sans
                        font.pixelSize: Theme.fontSmall
                    }
                }
            }

            // Settings.
            Card {
                Layout.fillWidth: true
                title: "Voice and input"

                ColumnLayout {
                    width: parent.width
                    spacing: 12

                    Segmented {
                        Layout.fillWidth: true
                        current: app.ttsEngine
                        options: [
                            { value: "piper", label: "Piper" },
                            { value: "sapi",  label: "SAPI" },
                            { value: "off",   label: "Muted" }
                        ]
                        onPicked: (value) => app.setTtsEngine(value)
                    }

                    SliderRow {
                        Layout.fillWidth: true
                        label: "Rate"
                        from: 0.6
                        to: 1.6
                        stepSize: 0.05
                        value: app.ttsRate
                        onValueChanged: (v) => app.setRate(v)
                    }

                    SliderRow {
                        Layout.fillWidth: true
                        label: "Volume"
                        from: 0.0
                        to: 1.0
                        stepSize: 0.05
                        value: app.ttsVolume
                        onValueChanged: (v) => app.setVolume(v)
                    }

                    SliderRow {
                        Layout.fillWidth: true
                        label: "Threshold"
                        from: 0.30
                        to: 0.95
                        stepSize: 0.01
                        value: app.minConfidence
                        onValueChanged: (v) => app.setMinConfidence(v)
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.gap
                        ToggleRow {
                            Layout.fillWidth: true
                            label: "Speak automatically"
                            checked: app.autoSpeak
                            onToggled: app.setAutoSpeak(checked)
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.gap
                        ToggleRow {
                            Layout.fillWidth: true
                            label: "Mirror preview"
                            checked: app.mirrored
                            onToggled: app.setMirrored(checked)
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Theme.gap
                        ToggleRow {
                            Layout.fillWidth: true
                            label: "Hand skeleton"
                            checked: app.showSkeleton
                            onToggled: app.setSkeleton(checked)
                        }
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------- clip picker

    FileDialog {
        id: clipDialog
        title: "Open a sign language clip"
        nameFilters: ["Video files (*.mp4 *.avi *.mov *.mkv)", "All files (*)"]
        onAccepted: app.setVideoPath(selectedFile.toString().replace("file:///", ""))
    }

    // ------------------------------------------------------------------ wire

    Connections {
        target: app
        function onErrorRaised(message) { console.warn("SignBridge:", message) }
    }

    Shortcut {
        sequence: "Space"
        onActivated: app.toggleRunning()
    }
    Shortcut {
        sequence: "Ctrl+Return"
        onActivated: app.flushPhrase()
    }
    Shortcut {
        sequence: "Ctrl+R"
        onActivated: app.repeatLast()
    }
}
