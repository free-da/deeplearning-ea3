
import { tokensToSequences, padSequences } from './tokenizer.js';
import { prepareLanguageModelData } from './lm-preprocessing.js';
import { createLanguageModel } from './model.js';

/**
 * Trainiert ein Sprachmodell (Next Word Prediction)
 *
 * @param {Array[]} tokenGroups - Array von Sätzen mit Tokens
 * @param {Object} vocab - Token->ID-Vokabular
 * @param {number} maxLen - Maximale Eingabesequenzlänge
 * @param {number} embeddingDim - Dimension der Embeddings
 * @param {number} lstmUnits - Anzahl LSTM-Einheiten
 * @param {number} epochs - Anzahl der Trainings-Epochen
 * @param {number} batchSize - Größe eines Batches
 * @returns {tf.LayersModel} Das trainierte Modell
 */
export async function trainLanguageModel({
                                             tokenGroups,
                                             vocab,
                                             maxLen,
                                             embeddingDim,
                                             lstmUnits,
                                             epochs,
                                             batchSize,
                                         }) {
    console.log("📦 Training mit Parametern:", { maxLen, embeddingDim, lstmUnits, epochs, batchSize });

    // Schritt 1: In Sequenzen umwandeln
    const tokenIds = tokensToSequences(tokenGroups, vocab);

    // Schritt 2: Trainingsdaten (X/y) erzeugen
    const { X, y } = prepareLanguageModelData(tokenIds, Object.keys(vocab).length);

    // Schritt 3: Padding & Tensoren erstellen
    const X_padded = tf.tensor2d(padSequences(X, maxLen));
    const yTensor = tf.tensor1d(y, 'int32');

    // Schritt 4: Modell erstellen mit übergebenen Parametern
    const model = createLanguageModel(Object.keys(vocab).length, maxLen, embeddingDim, lstmUnits);
    model.summary();

    console.log("X_padded shape:", X_padded.shape); // sollte [n, maxLen] sein
    console.log("maxLen erwartet:", maxLen);

    // Schritt 5: Modell trainieren
    await model.fit(X_padded, yTensor, {
        epochs,
        batchSize,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) =>
                console.log(`📈 Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}`),
        },
    });

    // Optional: Modell speichern
    await model.save('downloads://trained-lm');
    console.log("✅ Modell gespeichert");

    return model;
}
