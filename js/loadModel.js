
/**
 * Lädt ein gespeichertes Sprachmodell
 * @param {string} path - Pfad zur .json Datei
 * @returns {tf.LayersModel}
 */
export async function loadTrainedModel(path = './trained-lm') {
    const model = await tf.loadLayersModel(path);
    console.log("✅ Modell geladen");
    // await model.save('downloads://' + generateFilename());
    // console.log("Modell gespeichert")
    return model;
}

function generateFilename(prefix = "model") {
    const now = new Date();

    const pad = (n) => n.toString().padStart(2, '0');

    const timestamp = [
        now.getFullYear(),
        pad(now.getMonth() + 1),
        pad(now.getDate()),
        pad(now.getHours()),
        pad(now.getMinutes()),
        pad(now.getSeconds())
    ].join('-');

    return `${prefix}-${timestamp}`;
}
