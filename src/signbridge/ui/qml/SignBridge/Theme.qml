pragma Singleton
import QtQuick

/*
 * Nocturne: a dark, high contrast palette tuned so the camera image stays the
 * brightest thing on screen. Everything else recedes.
 */
QtObject {
    id: theme

    // Surfaces, from deepest to nearest.
    readonly property color bg:          "#06070D"
    readonly property color bgLift:      "#0B0D17"
    readonly property color surface:     "#11142339"
    readonly property color surface2:    "#161A2B"
    readonly property color surface3:    "#1E2338"
    readonly property color stroke:      "#242A40"
    readonly property color strokeBright:"#39415F"

    // Text.
    readonly property color text:        "#E9EDFB"
    readonly property color textDim:     "#98A3C0"
    readonly property color textFaint:   "#5D6785"

    // Accents.
    readonly property color accent:      "#5AE0FF"
    readonly property color accentDim:   "#1B6F84"
    readonly property color violet:      "#A78BFA"
    readonly property color good:        "#4ADE80"
    readonly property color warn:        "#FBBF24"
    readonly property color bad:         "#FB7185"

    function accentFor(value) {
        if (value >= 0.75) return good
        if (value >= 0.5) return accent
        if (value >= 0.3) return warn
        return bad
    }

    // Metrics.
    readonly property int radius:      18
    readonly property int radiusSmall: 12
    readonly property int pad:         20
    readonly property int gap:         14

    readonly property int fontDisplay: 44
    readonly property int fontTitle:   19
    readonly property int fontBody:    14
    readonly property int fontSmall:   12
    readonly property int fontMicro:   11

    readonly property string mono: "Consolas, 'Cascadia Mono', 'SF Mono', monospace"
    readonly property string sans: "Segoe UI Variable Display, 'Segoe UI', Inter, system-ui, sans-serif"

    // Motion.
    readonly property int fast: 120
    readonly property int normal: 220
    readonly property int slow: 420
}
