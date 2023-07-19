//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["S7", "S8", "S9", "F1", "F2"],
        }],
        output: {
            bands: 5,
            sampleType: "FLOAT32"
        }
    };
}
function evaluatePixel(sample) {
    return [
    sample.S7,
    sample.S8,
    sample.S9,
    sample.F1,
    sample.F2
    ];
}