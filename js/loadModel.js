
/**
 * Lädt ein gespeichertes Sprachmodell
 * @param {string} path - Pfad zur .json Datei
 * @returns {tf.LayersModel}
 */
export async function loadTrainedModel(path = 'trained-lm.json') {
    const model = await tf.loadLayersModel(path);
    console.log("✅ Modell geladen");
    return model;
}
