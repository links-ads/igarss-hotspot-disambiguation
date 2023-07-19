//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["S1", "S2", "S3", "S4", "S5", "S6"],
        }],
        output: {
            bands: 6,
            sampleType: "FLOAT32"
        }
    };
}
function evaluatePixel(sample) {
    return [sample.S1,
    sample.S2,
    sample.S3,
    sample.S4,
    sample.S5,
    sample.S6
    ];
}