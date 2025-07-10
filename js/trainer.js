import { tokensToSequences, padSequences } from './tokenizer.js';
import { prepareLanguageModelData } from './lm-preprocessing.js';
import { createLanguageModel } from './model.js';

/**
 * Trainiert ein Sprachmodell (Next Word Prediction)
 *
 * @param {Array[]} tokenGroups - Array von Sätzen mit Tokens
 * @param {Object} vocab - Token->ID-Vokabular (sollte <PAD>=0, <UNK>=1 enthalten)
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

    // Schritt 1: Token-Gruppen in Integer-Sequenzen übersetzen
    const tokenIds = tokensToSequences(tokenGroups, vocab);
    console.log("Beispiel tokenIds:", tokenIds.slice(0, 3));

    // Schritt 2: Trainingsdaten (X/y) erzeugen
    const { X, y } = prepareLanguageModelData(tokenIds, maxLen, vocab);
    const X_padded = tf.tensor2d(X);
    const yTensor = tf.tensor1d(y, 'int32');

    // Schritt 4: Modell erstellen
    const model = createLanguageModel(Object.keys(vocab).length, maxLen, embeddingDim, lstmUnits);
    model.summary();

    console.log("X_padded shape:", X_padded.shape); // [Anzahl Samples, maxLen]

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
