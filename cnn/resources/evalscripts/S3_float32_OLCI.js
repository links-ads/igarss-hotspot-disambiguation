//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20", "B21"],
        }],
        output: {
            bands: 21,
            sampleType: "FLOAT32"
        }
    };
}
function evaluatePixel(sample) {
    return [sample.B01,
    sample.B02,
    sample.B03,
    sample.B04,
    sample.B05,
    sample.B06,
    sample.B07,
    sample.B08,
    sample.B8A,
    sample.B09,
    sample.B10,
    sample.B11,
    sample.B12,
    sample.B13,
    sample.B14,
    sample.B15,
    sample.B16,
    sample.B17,
    sample.B18,
    sample.B19,
    sample.B20,
    sample.B21
    ];
}