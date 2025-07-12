import { tokensToSequences, padSequences } from './tokenizer.js';
import { prepareLanguageModelData, removeCitationsFromTokenGroups } from './lm-preprocessing.js';
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
    const cleanedTokenGroups = removeCitationsFromTokenGroups(tokenGroups);
    const tokenIds = tokensToSequences(cleanedTokenGroups, vocab);

    // Schritt 2: Trainingsdaten (X/y) erzeugen
    const { X, y } = prepareLanguageModelData(tokenIds, maxLen, vocab);
    // === Analyse-Logs zum Training ===
    const UNK_ID = vocab['<UNK>'];
    let totalUnks = 0;
    let totalTokens = 0;

    X.forEach(seq => {
        totalUnks += seq.filter(id => id === UNK_ID).length;
        totalTokens += seq.length;
    });

    const unkRatio = (totalUnks / totalTokens) * 100;
    console.log(`📊 UNK-Anteil in Inputs: ${unkRatio.toFixed(2)} %`);

    const uniqueTargets = new Set(y);
    console.log(`📊 Anzahl einzigartiger Zielwörter: ${uniqueTargets.size} von ${Object.keys(vocab).length}`);
    console.log(`📊 Gesamtzahl Trainingsbeispiele: ${X.length}`);

    const X_padded = tf.tensor2d(X);
    const yTensor = tf.tensor1d(y, 'int32');

    // Schritt 4: Modell erstellen
    const model = createLanguageModel(Object.keys(vocab).length, maxLen, embeddingDim, lstmUnits);
    model.summary();

    console.log("X_padded shape:", X_padded.shape); // [Anzahl Samples, maxLen]
    // Logge 5 zufällige Trainingsbeispiele
    const id2word = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));

    for (let i = 0; i < 5; i++) {
        const seqIndex = Math.floor(Math.random() * X.length);
        const inputTokens = X[seqIndex];
        const targetTokenId = y[seqIndex];

        const inputWords = inputTokens.map(id => id2word[id] || '<UNK>');
        const targetWord = id2word[targetTokenId] || '<UNK>';

        console.log(`📚 Beispiel ${i + 1}`);
        console.log("📥 Input:", inputWords.join(' '));
        console.log("🎯 Target:", targetWord);
    }

    // Schritt 5: Modell trainieren
    await model.fit(X_padded, yTensor, {
        epochs,
        batchSize,
        shuffle: true,
        callbacks: [
            {
                onBatchEnd: async (batch, logs) => {
                    if (batch % 25 === 0) {
                        console.log(`🧮 Batch ${batch}: Loss = ${logs.loss.toFixed(4)}`);
                    }
                },
                onEpochEnd: async (epoch, logs) => {
                    console.log(`📈 Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)}`);
                    if ((epoch + 1) % 2 === 0) {
                        await model.save('indexeddb://checkpoint');
                    }
                }
            }
        ]
    });
    await tf.nextFrame(); // Lässt den Browser „atmen“, reduziert Sync-Probleme


    // Optional: Modell speichern
    await model.save('downloads://trained-lm');
    console.log("✅ Modell gespeichert");

    return model;
}
